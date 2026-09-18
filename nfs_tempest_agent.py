"""
NFS & OpenStack Test Agent
==========================
Two AI agents in one file:

  1. Connectathon NFS Agent
     - Runs basic/general/special/lock test suites
     - Compares results across NFSv3, NFSv4, NFSv4.1 and mount points
     - Produces a pass/fail/regression report

  2. OpenStack Tempest Agent
     - Runs compute/network/storage/identity/image/object test groups
     - Compares results across two environment configs
     - Surfaces regressions, improvements, and failed test names

Requirements:
    pip install -r requirements.txt

    # Connectathon (Linux):
    git clone https://github.com/linux-nfs/connectathon /opt/connectathon
    cd /opt/connectathon && make

    # Tempest (OpenStack):
    pip install tempest
    export TEMPEST_PATH=/opt/tempest

API key (free, no credit card):
    export GOOGLE_API_KEY="AIza..."

Dry-run (no tools, no API key needed):
    export DRY_RUN=true
    python nfs_tempest_agent.py

Author : vytasta — https://vytasta.github.io/uberzunn/agents.html
License: MIT
"""

import os
import json
import time
import functools
import socket
import threading
from typing import TypedDict
from datetime import datetime

import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS, TOOL_MAP

# ─────────────────────────────────────────────────────────────
# AGENTLEASE CONTROL-PLANE WIRING
# Set these to register this run as a data plane against a real
# AgentLease lease and heartbeat while the agents run. If
# CONTROL_PLANE_URL / PAIRING_TOKEN aren't set, this is a no-op.
#
#   export CONTROL_PLANE_URL="https://your-control-plane.run.app"
#   export PAIRING_TOKEN="uz_live_..."
#   export DATA_PLANE_NAME="my-test-host"   # optional, defaults to hostname
#   export DATA_PLANE_ENV="bare-metal"      # optional
# ─────────────────────────────────────────────────────────────

CONTROL_PLANE_URL = os.environ.get("CONTROL_PLANE_URL", "").rstrip("/")
PAIRING_TOKEN     = os.environ.get("PAIRING_TOKEN", "")
DATA_PLANE_NAME   = os.environ.get("DATA_PLANE_NAME", socket.gethostname())
DATA_PLANE_ENV    = os.environ.get("DATA_PLANE_ENV", "other")
HEARTBEAT_SECONDS = int(os.environ.get("HEARTBEAT_SECONDS", "30"))

_heartbeat_stop = threading.Event()


def _control_plane_enabled():
    return bool(CONTROL_PLANE_URL and PAIRING_TOKEN)


def register_data_plane():
    if not _control_plane_enabled():
        return None
    try:
        resp = requests.post(
            f"{CONTROL_PLANE_URL}/dataplanes/register",
            json={"pairing_token": PAIRING_TOKEN, "name": DATA_PLANE_NAME, "environment": DATA_PLANE_ENV},
            timeout=10,
        )
        resp.raise_for_status()
        data_plane_id = resp.json()["id"]
        print(f"[control-plane] registered as data plane {data_plane_id} ({DATA_PLANE_NAME})")
        return data_plane_id
    except Exception as e:
        print(f"[control-plane] registration failed, continuing without it: {e}")
        return None


def _heartbeat_loop(data_plane_id):
    while not _heartbeat_stop.wait(HEARTBEAT_SECONDS):
        try:
            requests.post(
                f"{CONTROL_PLANE_URL}/dataplanes/{data_plane_id}/heartbeat",
                json={"pairing_token": PAIRING_TOKEN},
                timeout=10,
            )
        except Exception as e:
            print(f"[control-plane] heartbeat failed (will retry): {e}")


def start_heartbeat(data_plane_id):
    if not data_plane_id:
        return None
    t = threading.Thread(target=_heartbeat_loop, args=(data_plane_id,), daemon=True)
    t.start()
    print(f"[control-plane] heartbeating every {HEARTBEAT_SECONDS}s")
    return t


def stop_heartbeat():
    _heartbeat_stop.set()
from tools.connectathon_tools import CONNECTATHON_TOOLS
from tools.tempest_tools import TEMPEST_TOOLS

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DRY_RUN          = os.environ.get("DRY_RUN", "false").lower() == "true"
THROTTLE_SECONDS = 5
_last_call_time  = 0.0

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

if DRY_RUN:
    print("=" * 65)
    print("DRY-RUN MODE — simulated results, no API key needed")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# RATE LIMIT PROTECTION
# ─────────────────────────────────────────────────────────────

def throttle():
    global _last_call_time
    if DRY_RUN:
        return
    elapsed = time.time() - _last_call_time
    if elapsed < THROTTLE_SECONDS:
        wait = THROTTLE_SECONDS - elapsed
        print(f"  [throttle] waiting {wait:.1f}s...")
        time.sleep(wait)
    _last_call_time = time.time()


def rate_limited(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        throttle()
        return fn(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────
# AGENT STATES
# ─────────────────────────────────────────────────────────────

class ConnectathonState(TypedDict):
    messages:       list
    configs:        list    # list of generated configs (one per version/mount)
    raw_outputs:    list    # raw output per run
    parsed_results: list    # parsed result per run
    comparison:     dict    # final comparison
    final_answer:   str
    error:          str


class TempestState(TypedDict):
    messages:       list
    configs:        list    # list of generated configs (one per env config)
    raw_outputs:    list    # raw output per run
    parsed_results: list    # parsed result per run
    comparison:     dict    # final comparison
    final_answer:   str
    error:          str


# ─────────────────────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────────────────────

CTHON_SYSTEM = """You are an expert NFS and storage engineer AI agent.

Given a prompt about running Connectathon NFS tests, you will:
1. Call generate_connectathon_config for EACH NFS version or mount point specified.
   Each config gets its own label (e.g. "NFSv3", "NFSv4", "NFSv4.1" or the mount path).
2. Call run_connectathon for each config — one at a time.
3. Call parse_connectathon_output for each raw output, passing the label.
4. Once ALL runs are parsed, call compare_connectathon_results with a JSON array
   of ALL parsed results.
5. Stop — the agent framework will generate the final report.

Call ONE tool at a time. Never skip a step.
Default test suites if not specified: b,g,s,l (all four).
Default timeout: 300 seconds per suite.
"""

TEMPEST_SYSTEM = """You are an expert OpenStack engineer AI agent.

Given a prompt about running OpenStack Tempest tests, you will:
1. Call generate_tempest_config for EACH environment config specified.
   Each config gets its own label (e.g. "prod", "staging", "config-A", "config-B").
2. Call run_tempest for each config — one at a time.
3. Call parse_tempest_output for each raw output, passing the label.
4. Once ALL runs are parsed, call compare_tempest_results with a JSON array
   of ALL parsed results.
5. Stop — the agent framework will generate the final report.

Call ONE tool at a time. Never skip a step.
Default test groups if not specified: compute,network,storage,identity,image.
"""


def _make_llm(tools_list):
    if DRY_RUN:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    ).bind_tools(tools_list)


def _make_llm_plain():
    if DRY_RUN:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )


CTHON_LLM  = _make_llm(CONNECTATHON_TOOLS)
TEMPEST_LLM = _make_llm(TEMPEST_TOOLS)
LLM_PLAIN  = _make_llm_plain()


# ─────────────────────────────────────────────────────────────
# DRY-RUN SEQUENCES
# ─────────────────────────────────────────────────────────────

_CTHON_DRY_SEQUENCE = [
    ("generate_connectathon_config", {"server": "nfs-server-01", "mount_point": "/exports/vol1", "nfs_version": "3",   "test_suites": "b,g,s,l", "test_dir": "/mnt/nfstest_v3"}),
    ("run_connectathon",             {"config_json": '{"server":"nfs-server-01","mount_point":"/exports/vol1","nfs_version":"3","test_suites":["b","g","s","l"],"test_dir":"/mnt/nfstest_v3","dry_run":true}'}),
    ("parse_connectathon_output",    {"raw_output": "placeholder_v3", "label": "NFSv3"}),
    ("generate_connectathon_config", {"server": "nfs-server-01", "mount_point": "/exports/vol1", "nfs_version": "4",   "test_suites": "b,g,s,l", "test_dir": "/mnt/nfstest_v4"}),
    ("run_connectathon",             {"config_json": '{"server":"nfs-server-01","mount_point":"/exports/vol1","nfs_version":"4","test_suites":["b","g","s","l"],"test_dir":"/mnt/nfstest_v4","dry_run":true}'}),
    ("parse_connectathon_output",    {"raw_output": "placeholder_v4", "label": "NFSv4"}),
    ("generate_connectathon_config", {"server": "nfs-server-01", "mount_point": "/exports/vol1", "nfs_version": "4.1", "test_suites": "b,g,s,l", "test_dir": "/mnt/nfstest_v41"}),
    ("run_connectathon",             {"config_json": '{"server":"nfs-server-01","mount_point":"/exports/vol1","nfs_version":"4.1","test_suites":["b","g","s","l"],"test_dir":"/mnt/nfstest_v41","dry_run":true}'}),
    ("parse_connectathon_output",    {"raw_output": "placeholder_v41", "label": "NFSv4.1"}),
    ("compare_connectathon_results", {}),
]

_TEMPEST_DRY_SEQUENCE = [
    ("generate_tempest_config", {"auth_url": "http://controller-prod:5000/v3",    "username": "admin", "password": "secret", "project_name": "admin", "config_label": "prod",    "test_groups": "compute,network,storage,identity,image"}),
    ("run_tempest",             {"config_json": '{"label":"prod","auth_url":"http://controller-prod:5000/v3","test_groups":["compute","network","storage","identity","image"],"concurrency":4,"run_timeout":600,"dry_run":true}'}),
    ("parse_tempest_output",    {"raw_output": "placeholder_prod",    "label": "prod"}),
    ("generate_tempest_config", {"auth_url": "http://controller-staging:5000/v3", "username": "admin", "password": "secret", "project_name": "admin", "config_label": "staging", "test_groups": "compute,network,storage,identity,image"}),
    ("run_tempest",             {"config_json": '{"label":"staging","auth_url":"http://controller-staging:5000/v3","test_groups":["compute","network","storage","identity","image"],"concurrency":4,"run_timeout":600,"dry_run":true}'}),
    ("parse_tempest_output",    {"raw_output": "placeholder_staging", "label": "staging"}),
    ("compare_tempest_results", {}),
]

_cthon_step   = 0
_tempest_step = 0


class _DryMsg(AIMessage):
    def __init__(self, name, args, call_id):
        super().__init__(
            content=f"[DRY-RUN] {name}",
            tool_calls=[{"name": name, "args": args, "id": call_id}],
        )

def _make_done_msg():
    return AIMessage(content="[DRY-RUN] sequence complete", tool_calls=[])


# ─────────────────────────────────────────────────────────────
# CONNECTATHON GRAPH NODES
# ─────────────────────────────────────────────────────────────

@rate_limited
def _cthon_llm_invoke(messages):
    return CTHON_LLM.invoke(messages)


def cthon_planner_node(state: ConnectathonState) -> dict:
    global _cthon_step
    if DRY_RUN:
        if _cthon_step >= len(_CTHON_DRY_SEQUENCE) - 1:
            return {"messages": state["messages"] + [_make_done_msg()]}
        name, args = _CTHON_DRY_SEQUENCE[_cthon_step]
        _cthon_step += 1
        print(f"  [dry-run cthon] step {_cthon_step}: {name}")
        time.sleep(0.2)
        return {"messages": state["messages"] + [_DryMsg(name, args, f"cthon-{_cthon_step}")]}

    messages = [{"role": "system", "content": CTHON_SYSTEM}, *state["messages"]]
    response = _cthon_llm_invoke(messages)
    return {"messages": state["messages"] + [response]}


def cthon_tool_executor_node(state: ConnectathonState) -> dict:
    global _cthon_step
    last    = state["messages"][-1]
    updates = {"configs": list(state.get("configs", [])),
               "raw_outputs": list(state.get("raw_outputs", [])),
               "parsed_results": list(state.get("parsed_results", []))}
    new_msgs = list(state["messages"])

    for tc in last.tool_calls:
        fn = TOOL_MAP.get(tc["name"])

        # In dry-run, inject real config/output for parse/compare steps
        args = tc["args"]
        if DRY_RUN:
            args = _inject_dry_run_args(tc["name"], args, updates)

        result = fn.invoke(args) if fn else json.dumps({"error": f"Unknown tool: {tc['name']}"})
        new_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        if tc["name"] == "generate_connectathon_config":
            updates["configs"].append(json.loads(result))
        elif tc["name"] == "run_connectathon":
            updates["raw_outputs"].append(result)
        elif tc["name"] == "parse_connectathon_output":
            try:    updates["parsed_results"].append(json.loads(result))
            except: updates["parsed_results"].append({"raw": result})
        elif tc["name"] == "compare_connectathon_results":
            try:    updates["comparison"] = json.loads(result)
            except: updates["comparison"] = {"raw": result}

    return {"messages": new_msgs, **updates}


def _inject_dry_run_args(tool_name, args, state_updates):
    """Inject real state data into dry-run tool args for parse/compare steps."""
    if tool_name == "run_connectathon" and state_updates.get("configs"):
        last_config = state_updates["configs"][-1]
        last_config["dry_run"] = True
        return {"config_json": json.dumps(last_config)}
    elif tool_name == "parse_connectathon_output" and state_updates.get("raw_outputs"):
        label = args.get("label", f"run-{len(state_updates['parsed_results'])+1}")
        return {"raw_output": state_updates["raw_outputs"][-1], "label": label}
    elif tool_name == "run_tempest" and state_updates.get("configs"):
        last_config = state_updates["configs"][-1]
        last_config["dry_run"] = True
        return {"config_json": json.dumps(last_config)}
    elif tool_name == "parse_tempest_output" and state_updates.get("raw_outputs"):
        label = args.get("label", f"run-{len(state_updates['parsed_results'])+1}")
        return {"raw_output": state_updates["raw_outputs"][-1], "label": label}
    elif tool_name in ("compare_connectathon_results", "compare_tempest_results"):
        parsed = state_updates.get("parsed_results", [])
        return {"results_json_list": json.dumps(parsed)}
    return args


@rate_limited
def _cthon_report_invoke(prompt):
    return LLM_PLAIN.invoke([{"role": "user", "content": prompt}])


def cthon_report_node(state: ConnectathonState) -> dict:
    comparison = state.get("comparison", {})
    parsed     = state.get("parsed_results", [])

    if DRY_RUN:
        answer = _build_cthon_report(comparison, parsed)
        return {"final_answer": answer}

    prompt = f"""
You are an expert NFS engineer. Produce a clear Connectathon test report.

Parsed results per NFS version/mount:
{json.dumps(parsed, indent=2)}

Comparison:
{json.dumps(comparison, indent=2)}

Format EXACTLY as:

CONNECTATHON NFS TEST REPORT
==============================
Date: [today]

RESULTS SUMMARY
---------------
Version/Mount  | Overall | Passed | Failed | Skipped | Errors
---------------+---------+--------+--------+---------+-------
[one row per run]

PER-SUITE BREAKDOWN
-------------------
Suite    | [label1] P/F | [label2] P/F | [label3] P/F
---------+--------------+--------------+--------------
basic    | ...          | ...          | ...
general  | ...          | ...          | ...
special  | ...          | ...          | ...
lock     | ...          | ...          | ...

REGRESSIONS
-----------
[list any suites with more failures vs baseline, or "None detected"]

IMPROVEMENTS
------------
[list any suites with fewer failures vs baseline, or "None detected"]

ANALYSIS
--------
[3-4 sentences comparing NFS versions, noting which performs best and any
known NFSv3 vs NFSv4 behavioural differences in the failures]

RECOMMENDATION
--------------
[Which NFS version passed cleanest and is recommended for production]
"""
    response = _cthon_report_invoke(prompt)
    return {"final_answer": response.content}


def _build_cthon_report(comparison, parsed) -> str:
    labels   = [r.get("label", f"run-{i}") for i, r in enumerate(parsed)]
    summary  = comparison.get("summary", [])
    suite_diff = comparison.get("suite_diff", {})
    regressions = comparison.get("regressions", [])
    improvements = comparison.get("improvements", [])

    rows = ""
    for s in summary:
        rows += f"\n{s['label']:<14} | {s['overall']:<7} | {s['passed']:<6} | {s['failed']:<6} | {s['skipped']:<7} | {s['errors']}"

    suite_rows = ""
    for suite, data in suite_diff.items():
        row = f"\n{suite:<8}"
        for d in data:
            row += f" | {d['passed']}P/{d['failed']}F"
        suite_rows += row

    reg_text = "\n".join([
        f"  [{r['suite']}] {r['baseline']} → {r['label']}: +{r['delta']} failures"
        for r in regressions
    ]) or "  None detected"

    imp_text = "\n".join([
        f"  [{i['suite']}] {i['baseline']} → {i['label']}: -{i['delta']} failures"
        for i in improvements
    ]) or "  None detected"

    return f"""CONNECTATHON NFS TEST REPORT  [DRY-RUN SIMULATION]
=====================================================
Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Versions: {', '.join(labels)}

RESULTS SUMMARY
---------------
Version/Mount  | Overall | Passed | Failed | Skipped | Errors
---------------+---------+--------+--------+---------+-------{rows}

PER-SUITE BREAKDOWN
-------------------
Suite    | {' | '.join(labels)}
---------+{'-'*40}{suite_rows}

REGRESSIONS
-----------
{reg_text}

IMPROVEMENTS
------------
{imp_text}

ANALYSIS
--------
NFSv3 and NFSv4 show near-identical results on basic and general suites.
The special suite setuid/setgid failure on NFSv4 is expected behaviour —
NFSv4 maps UID 0 to nobody by default (root squash) which blocks setuid tests.
Lock suite passes cleanly across all versions confirming NLM/NFS4-locking health.

RECOMMENDATION
--------------
NFSv4.1 is recommended for production — cleanest lock test results and
pNFS-ready. NFSv3 is acceptable for legacy workloads. Investigate the
special suite setuid failure on NFSv4 if running privileged container workloads.
"""


# ─────────────────────────────────────────────────────────────
# TEMPEST GRAPH NODES
# ─────────────────────────────────────────────────────────────

@rate_limited
def _tempest_llm_invoke(messages):
    return TEMPEST_LLM.invoke(messages)


def tempest_planner_node(state: TempestState) -> dict:
    global _tempest_step
    if DRY_RUN:
        if _tempest_step >= len(_TEMPEST_DRY_SEQUENCE) - 1:
            return {"messages": state["messages"] + [_make_done_msg()]}
        name, args = _TEMPEST_DRY_SEQUENCE[_tempest_step]
        _tempest_step += 1
        print(f"  [dry-run tempest] step {_tempest_step}: {name}")
        time.sleep(0.2)
        return {"messages": state["messages"] + [_DryMsg(name, args, f"tempest-{_tempest_step}")]}

    messages = [{"role": "system", "content": TEMPEST_SYSTEM}, *state["messages"]]
    response = _tempest_llm_invoke(messages)
    return {"messages": state["messages"] + [response]}


def tempest_tool_executor_node(state: TempestState) -> dict:
    last    = state["messages"][-1]
    updates = {"configs": list(state.get("configs", [])),
               "raw_outputs": list(state.get("raw_outputs", [])),
               "parsed_results": list(state.get("parsed_results", []))}
    new_msgs = list(state["messages"])

    for tc in last.tool_calls:
        fn   = TOOL_MAP.get(tc["name"])
        args = tc["args"]
        if DRY_RUN:
            args = _inject_dry_run_args(tc["name"], args, updates)

        result = fn.invoke(args) if fn else json.dumps({"error": f"Unknown: {tc['name']}"})
        new_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        if tc["name"] == "generate_tempest_config":
            updates["configs"].append(json.loads(result))
        elif tc["name"] == "run_tempest":
            updates["raw_outputs"].append(result)
        elif tc["name"] == "parse_tempest_output":
            try:    updates["parsed_results"].append(json.loads(result))
            except: updates["parsed_results"].append({"raw": result})
        elif tc["name"] == "compare_tempest_results":
            try:    updates["comparison"] = json.loads(result)
            except: updates["comparison"] = {"raw": result}

    return {"messages": new_msgs, **updates}


@rate_limited
def _tempest_report_invoke(prompt):
    return LLM_PLAIN.invoke([{"role": "user", "content": prompt}])


def tempest_report_node(state: TempestState) -> dict:
    comparison = state.get("comparison", {})
    parsed     = state.get("parsed_results", [])

    if DRY_RUN:
        answer = _build_tempest_report(comparison, parsed)
        return {"final_answer": answer}

    prompt = f"""
You are a senior OpenStack engineer. Produce a clear Tempest test comparison report.

Parsed results per config:
{json.dumps(parsed, indent=2)}

Comparison:
{json.dumps(comparison, indent=2)}

Format EXACTLY as:

OPENSTACK TEMPEST TEST REPORT
==============================
Date: [today]

RESULTS SUMMARY
---------------
Config   | Overall | Pass% | Passed | Failed | Skipped | Duration
---------+---------+-------+--------+--------+---------+---------
[one row per config]

PER-GROUP BREAKDOWN
-------------------
Group     | [config1] P/F | [config2] P/F
----------+---------------+---------------
compute   | ...           | ...
network   | ...           | ...
storage   | ...           | ...
identity  | ...           | ...
image     | ...           | ...

REGRESSIONS (new failures in config2 vs config1)
-------------------------------------------------
[list group and test name, or "None detected"]

IMPROVEMENTS (fixed in config2 vs config1)
------------------------------------------
[list group, or "None detected"]

FAILED TESTS
------------
[list each failed test by config]

ANALYSIS
--------
[3-4 sentences comparing the two configs, noting which services regressed
and likely root causes based on the test names]

RECOMMENDATION
--------------
[Which config is more stable and what to investigate before promoting to prod]
"""
    response = _tempest_report_invoke(prompt)
    return {"final_answer": response.content}


def _build_tempest_report(comparison, parsed) -> str:
    summary      = comparison.get("summary", [])
    group_diff   = comparison.get("group_diff", {})
    regressions  = comparison.get("regressions", [])
    improvements = comparison.get("improvements", [])
    failed_tests = comparison.get("failed_tests", {})

    rows = ""
    for s in summary:
        rows += (f"\n{s['label']:<8} | {s['overall']:<7} | {s['pass_rate']:<5}% "
                 f"| {s['passed']:<6} | {s['failed']:<6} | {s['skipped']:<7} | {s['duration']}s")

    group_rows = ""
    for group, data in group_diff.items():
        row = f"\n{group:<9}"
        for d in data:
            row += f" | {d['passed']}P/{d['failed']}F"
        group_rows += row

    reg_text = "\n".join([
        f"  [{r['group']}] +{r['delta']} failures in {r['label']} vs {r['baseline']}"
        + (f"\n    New: {', '.join(r['new_failures'])}" if r.get('new_failures') else "")
        for r in regressions
    ]) or "  None detected"

    imp_text = "\n".join([
        f"  [{i['group']}] -{i['delta']} failures in {i['label']} vs {i['baseline']}"
        for i in improvements
    ]) or "  None detected"

    ft_text = ""
    for label, tests in failed_tests.items():
        ft_text += f"\n  {label}:\n"
        ft_text += "\n".join(f"    - {t}" for t in tests) if tests else "    None"

    return f"""OPENSTACK TEMPEST TEST REPORT  [DRY-RUN SIMULATION]
====================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTS SUMMARY
---------------
Config   | Overall | Pass% | Passed | Failed | Skipped | Duration
---------+---------+-------+--------+--------+---------+---------{rows}

PER-GROUP BREAKDOWN
-------------------
Group     | {' | '.join([s['label'] for s in summary])}
----------+{'-'*40}{group_rows}

REGRESSIONS (new failures vs baseline)
---------------------------------------
{reg_text}

IMPROVEMENTS (fixed vs baseline)
---------------------------------
{imp_text}

FAILED TESTS
------------{ft_text}

ANALYSIS
--------
Staging shows 2 additional failures vs prod in compute (attach_volume) and
storage (clone_volume). The attach_volume failure likely indicates a Cinder/Nova
integration misconfiguration in staging. The clone_volume failure suggests the
staging Cinder backend does not support volume cloning or has a quota issue.
Network, identity, and image suites are clean across both configs.

RECOMMENDATION
--------------
Do NOT promote staging config to production yet. Investigate Cinder backend
configuration in staging (volume_type, backend driver, and quota settings)
and verify Nova-Cinder API endpoint connectivity before re-running Tempest.
"""


# ─────────────────────────────────────────────────────────────
# EDGE ROUTING
# ─────────────────────────────────────────────────────────────

def route_planner(state, done_key="comparison") -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tool"
    if state.get(done_key):
        return "report"
    return "report"


def route_after_tool(state, parsed_key="parsed_results",
                     comparison_key="comparison") -> str:
    if state.get(comparison_key):
        return "report"
    return "planner"


# ─────────────────────────────────────────────────────────────
# BUILD GRAPHS
# ─────────────────────────────────────────────────────────────

def build_connectathon_agent():
    graph = StateGraph(ConnectathonState)
    graph.add_node("planner",      cthon_planner_node)
    graph.add_node("execute_tool", cthon_tool_executor_node)
    graph.add_node("report",       cthon_report_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", lambda s: route_planner(s), {
        "execute_tool": "execute_tool",
        "report":       "report",
    })
    graph.add_conditional_edges("execute_tool", lambda s: route_after_tool(s), {
        "planner": "planner",
        "report":  "report",
    })
    graph.add_edge("report", END)
    return graph.compile(checkpointer=MemorySaver())


def build_tempest_agent():
    graph = StateGraph(TempestState)
    graph.add_node("planner",      tempest_planner_node)
    graph.add_node("execute_tool", tempest_tool_executor_node)
    graph.add_node("report",       tempest_report_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", lambda s: route_planner(s), {
        "execute_tool": "execute_tool",
        "report":       "report",
    })
    graph.add_conditional_edges("execute_tool", lambda s: route_after_tool(s), {
        "planner": "planner",
        "report":  "report",
    })
    graph.add_edge("report", END)
    return graph.compile(checkpointer=MemorySaver())


# ─────────────────────────────────────────────────────────────
# RUNNERS
# ─────────────────────────────────────────────────────────────

def run_connectathon_agent(prompt: str) -> dict:
    global _cthon_step
    _cthon_step = 0

    agent  = build_connectathon_agent()
    config = {"configurable": {"thread_id": f"cthon-{int(time.time())}"}}
    state  = {
        "messages": [HumanMessage(content=prompt)],
        "configs": [], "raw_outputs": [], "parsed_results": [],
        "comparison": {}, "final_answer": "", "error": "",
    }

    print("\n" + "=" * 65)
    print("CONNECTATHON NFS AGENT")
    print("=" * 65)
    print(f"Prompt : {prompt}")
    print(f"Mode   : {'DRY-RUN' if DRY_RUN else 'LIVE'}")
    print("=" * 65 + "\n")

    result = agent.invoke(state, config=config)
    print("\n" + "=" * 65)
    print(result.get("final_answer", ""))
    return result


def run_tempest_agent(prompt: str) -> dict:
    global _tempest_step
    _tempest_step = 0

    agent  = build_tempest_agent()
    config = {"configurable": {"thread_id": f"tempest-{int(time.time())}"}}
    state  = {
        "messages": [HumanMessage(content=prompt)],
        "configs": [], "raw_outputs": [], "parsed_results": [],
        "comparison": {}, "final_answer": "", "error": "",
    }

    print("\n" + "=" * 65)
    print("OPENSTACK TEMPEST AGENT")
    print("=" * 65)
    print(f"Prompt : {prompt}")
    print(f"Mode   : {'DRY-RUN' if DRY_RUN else 'LIVE'}")
    print("=" * 65 + "\n")

    result = agent.invoke(state, config=config)
    print("\n" + "=" * 65)
    print(result.get("final_answer", ""))
    return result


def save_report(name: str, result: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"{name}_{timestamp}.json")
    report = {
        "timestamp":      timestamp,
        "dry_run":        DRY_RUN,
        "parsed_results": result.get("parsed_results"),
        "comparison":     result.get("comparison"),
        "report":         result.get("final_answer"),
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    txt_path = path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        f.write(result.get("final_answer", ""))
    print(f"\n  Report saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DRY_RUN and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set.")
        print("Get a free key at https://aistudio.google.com")
        print("Or: export DRY_RUN=true")
        exit(1)

    data_plane_id = register_data_plane()
    start_heartbeat(data_plane_id)

    try:
        # ── Connectathon prompt ───────────────────────────────────
        cthon_prompt = (
            "Run Connectathon NFS tests (basic, general, special, lock) "
            "on nfs-server-01:/exports/vol1 comparing NFSv3, NFSv4, and NFSv4.1. "
            "Use /mnt/nfstest as the test directory. Timeout 300s per suite."
        )

        # ── Tempest prompt ────────────────────────────────────────
        tempest_prompt = (
            "Run OpenStack Tempest tests (compute, network, storage, identity, image) "
            "comparing prod config (http://controller-prod:5000/v3) vs "
            "staging config (http://controller-staging:5000/v3). "
            "Use admin credentials, 4 concurrent workers."
        )

        # Run both agents
        cthon_result   = run_connectathon_agent(cthon_prompt)
        tempest_result = run_tempest_agent(tempest_prompt)

        # Save reports
        save_report("connectathon", cthon_result)
        save_report("tempest",      tempest_result)

        print("\n" + "=" * 65)
        print("BOTH AGENTS COMPLETE — reports saved to reports/")
        print("=" * 65)
    finally:
        # Results still only go to local report files, not back to the
        # control plane — same open gap noted for the other agents.
        stop_heartbeat()
