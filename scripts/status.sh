#!/usr/bin/env bash
# Short status of the v2 run:  bash scripts/status.sh [log]   (default runs/v2_m1.log)
LOG="${1:-runs/v2_m1.log}"
echo "== stage: $(grep -E '^=== .* (start|done) ' "$LOG" 2>/dev/null | tail -1)"
echo "== last 15 lines of $LOG"
tail -15 "$LOG" 2>/dev/null
echo "== disk / RAM"
df -h ~ | tail -1
free -g | head -2
echo "== our python processes"
ps -u "$USER" -o pid,etime,%cpu,rss,cmd | grep -E "python|PID" | grep -v grep | cut -c1-150
