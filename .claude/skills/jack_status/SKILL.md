---
name: jack_status
description: Quick status check of NEXUS platform — terminals, dashboard, signals, git. Use for a fast overview without full supervisor mode.
allowed-tools: [Bash, Read]
---

# NEXUS Quick Status

Run these checks and report a concise summary:

```bash
cd "/Users/qtmobile/Desktop/Nexus - Quant Trading "

echo "=== GIT STATUS ==="
git log --oneline -3
git status -s | head -20

echo ""
echo "=== TERMINAL STATES ==="
python3 -c "
from nexus_quant.orchestration.terminal_state import get_dashboard_summary
import json
s = get_dashboard_summary()
print(f\"Running: {s['running']} | Stale: {s['stale']} | Dead: {s['dead']} | Blocked: {s['blocked']}\")
for t in s['terminals']:
    print(f\"  {t['terminal_id']}: {t['status']} ({t.get('age_human','?')}) - {t.get('task','?')}\")
for a in s['alerts']:
    print(f\"  {a}\")
" 2>/dev/null || echo "No terminal states found"

echo ""
echo "=== DASHBOARD ==="
curl -s http://localhost:8080/api/system_status 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Dashboard not running"

echo ""
echo "=== LATEST SIGNAL ==="
ls -lt artifacts/live/signal_*.json 2>/dev/null | head -1 || echo "No signals"
```

Report format:
- One-line summary per section
- Flag anything that needs attention
- Vietnamese OK
