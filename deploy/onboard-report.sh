#!/usr/bin/env bash
# onboard-report.sh — runs INSIDE a clean container, after provision-worker.sh, and reports
# what actually happened in a form a machine can judge.
#
# ⚠️⚠️ IT REPORTS OBSERVATIONS, NOT CONCLUSIONS. It does not decide whether onboarding
# succeeded — `onboardcheck.onboard_succeeded()` does that, outside, from this JSON. Same
# split as the join phases: the thing with side effects gathers, the pure thing judges.
#
# ⚠️ IT READS THE PROCESS TABLE, NOT THE INSTALLER'S LOG. An installer that PRINTS
# "--n-gpu-layers 99" has told you its intention; /proc tells you what happened. Those agree
# only when nothing went wrong — which is exactly the case that needed no test. An installer
# shipped `--n-gpu-layers 0` for weeks while printing the right thing, and every check that
# read the log agreed with it.
set -uo pipefail          # ⚠️ deliberately NOT -e: a failing step must still be REPORTED

PROVISION="${1:-/provision.sh}"
PORT="${NKS_PORT:-8080}"

bash "$PROVISION"
provision_rc=$?

pid=""; argv_json="null"; accel="null"; status="null"
# The worker daemon, if the installer started one.
pid="$(pgrep -f 'llama-.*-worker|worker\.py' | head -1 || true)"
if [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ]; then
  argv_json="$(tr '\0' '\n' < "/proc/$pid/cmdline" | python3 -c \
    'import json,sys; print(json.dumps([l for l in sys.stdin.read().split("\n") if l]))')"
fi
[ -n "${ACCEL:-}" ] && accel="\"$ACCEL\""
status="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
          "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)"
[ "$status" = "000" ] && status="null"

# ⚠️ Markers on the LAST line, and the reader scans from the END: a build log is long and a
# package name containing the marker string must not beat the real report.
printf '###NAKSHATRA-REPORT###{"exit_code":%d,"daemon_pid":%s,"running_argv":%s,"detected_accel":%s,"http_status":%s}###END###\n' \
  "$provision_rc" "${pid:-null}" "$argv_json" "$accel" "$status"
