"""Read the report a provisioning container emitted about its own run.

The container prints a long build log and then, on its last line, one JSON object between
`###NAKSHATRA-REPORT###` and `###END###`.

⚠️⚠️ SCAN FROM THE END. A provisioning log is thousands of lines of apt output, and a package
name or an echoed script line containing the marker string must not beat the real report at
the bottom. Taking the LAST marker pair is what makes that impossible rather than unlikely.

⚠️ "No report" is a DIFFERENT ANSWER from "the report says it failed", and the caller must be
able to tell them apart: no report means the container died before it could speak, which
usually points at the harness, not the installer — and telling an operator their installer is
broken when the harness is costs a day.

⚠️ Everything outside the markers is ignored entirely. It is untrusted output from a script
fetched over the network.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

START, END = "###NAKSHATRA-REPORT###", "###END###"

#: field -> the types it may legitimately hold (None always allowed).
FIELDS: Dict[str, tuple] = {
    "exit_code": (int,),
    "daemon_pid": (int,),
    "running_argv": (list,),
    "detected_accel": (str,),
    "http_status": (int,),
}


def parse_report(stdout: Any) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(stdout, str):
        return {}, ["container output is not text — no report could be read"]

    end = stdout.rfind(END)
    start = stdout.rfind(START, 0, end if end != -1 else len(stdout))
    if start == -1 or end == -1 or end <= start:
        return {}, ["the container produced no report — it died before it could report, "
                    "which usually points at the harness rather than the installer"]

    blob = stdout[start + len(START):end].strip()
    try:
        data = json.loads(blob)
    except (ValueError, RecursionError) as e:
        return {}, [f"the report between the markers is not valid JSON: {e}"]
    if not isinstance(data, dict):
        return {}, [f"the report is a {type(data).__name__}, not an object"]

    report: Dict[str, Any] = {}
    problems: List[str] = []
    for name, types in FIELDS.items():
        if name not in data:
            # ⚠️ Filled with None and NOTED, not rejected: a report missing one field is still
            # worth judging on the others, and `onboardcheck` treats None as a failure anyway.
            report[name] = None
            problems.append(f"report is missing '{name}'")
            continue
        v = data[name]
        if v is None or (isinstance(v, types) and not isinstance(v, bool)):
            report[name] = v
        else:
            report[name] = None
            problems.append(f"report field '{name}' is a {type(v).__name__}, expected "
                            f"{'/'.join(t.__name__ for t in types)}")
    extra = sorted(set(data) - set(FIELDS))
    if extra:
        # Not fatal — a newer container may report more. Note it so a version skew is visible.
        problems.append(f"report has unexpected field(s): {', '.join(extra)}")
    return report, problems
