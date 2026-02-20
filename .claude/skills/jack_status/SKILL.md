---
name: jack_status
description: Quick status check of NEXUS platform — terminals, dashboard, signals, git. Use for a fast overview without full supervisor mode.
allowed-tools: [Bash, Read]
---

# NEXUS Quick Status

```bash
# Works on any machine / any folder
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJECT_ROOT"

echo "=== PROJECT: $PROJECT_ROOT ==="

echo "=== GIT ==="
git log --oneline -3
git status -s | head -10

echo "=== TERMINALS ==="
python3 -c "
import sys; sys.path.insert(0, '.')
from nexus_quant.orchestration.terminal_state import get_dashboard_summary
s = get_dashboard_summary()
print(f'Running:{s[\"running\"]} Stale:{s[\"stale\"]} Dead:{s[\"dead\"]} Blocked:{s[\"blocked\"]}')
for t in s['terminals']:
    icon = {'running':'✅','stale':'⚠️','dead':'🔴','blocked':'🟡','completed':'✔️'}.get(t.get('status',''),'❓')
    print(f'  {icon} {t[\"terminal_id\"]}: {t[\"status\"]} ({t.get(\"age_human\",\"?\")}) — {t.get(\"task\",\"?\")}')
for a in s['alerts']: print(f'  {a}')
" 2>/dev/null || echo "No states"

echo "=== DASHBOARD ==="
curl -s http://localhost:8080/api/system_status 2>/dev/null | python3 -c "
import sys,json
try: d=json.load(sys.stdin); print(f'UP v{d.get(\"version\",\"?\")}')
except: print('DOWN')
" || echo "DOWN"

echo "=== LATEST SIGNAL ==="
ls -lt "$PROJECT_ROOT/artifacts/live/signal_"*.json 2>/dev/null | head -1 || echo "No signals"
```

Report: one-line per section, flag issues, Vietnamese OK.
