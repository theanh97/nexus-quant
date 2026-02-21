---
name: jack_status
description: Quick status check of NEXUS platform — ALL 3 projects, terminals, dashboard, models, signals, git. Flags constitution violations.
---

# NEXUS Quick Status

> **Obey CLAUDE.md (Constitution). Check ALL 3 projects, not just one.**

```bash
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJECT_ROOT"

echo "=== PROJECT: $PROJECT_ROOT ==="

echo "=== GIT ==="
git log --oneline -5
git status -s | head -10

echo "=== ALL 3 PROJECTS ==="
for proj in crypto_perps commodity_cta crypto_options; do
  echo "--- $proj ---"
  if [ -f "nexus_quant/projects/$proj/project.yaml" ]; then
    python3 -c "
import sys; sys.path.insert(0, '.')
try:
    from nexus_quant.projects import ProjectManifest
    from pathlib import Path
    m = ProjectManifest.from_file(Path('nexus_quant/projects/$proj/project.yaml'))
    print(f'  Enabled: {m.enabled} | Strategies: {len(m.strategies)} | Universe: {len(m.universe)}')
except Exception as e:
    print(f'  Error loading: {e}')
" 2>/dev/null || echo "  project.yaml exists but can't parse"
  else
    echo "  MISSING project.yaml!"
  fi
done

echo "=== TERMINALS ==="
python3 -c "
import sys; sys.path.insert(0, '.')
from nexus_quant.orchestration.terminal_state import get_dashboard_summary
s = get_dashboard_summary()
print(f'Running:{s[\"running\"]} Stale:{s[\"stale\"]} Dead:{s[\"dead\"]} Blocked:{s[\"blocked\"]}')
for t in s['terminals']:
    icon = {'running':'✅','stale':'⚠️','dead':'🔴','blocked':'🟡','completed':'✔️'}.get(t.get('status',''),'❓')
    print(f'  {icon} {t[\"terminal_id\"]}: {t[\"status\"]} ({t.get(\"age_human\",\"?\")}) — {t.get(\"task\",\"?\")}')
for a in s['alerts']: print(f'  ⚠️ {a}')
" 2>/dev/null || echo "No terminal states"

echo "=== DASHBOARD ==="
curl -s http://localhost:8080/api/system_status 2>/dev/null | python3 -c "
import sys,json
try: d=json.load(sys.stdin); print(f'UP v{d.get(\"version\",\"?\")}')
except: print('DOWN')
" || echo "DOWN"

echo "=== MODEL KEYS ==="
python3 -c "
import os
keys = {'GEMINI_API_KEY': bool(os.environ.get('GEMINI_API_KEY')),
        'ANTHROPIC_API_KEY': bool(os.environ.get('ANTHROPIC_API_KEY')),
        'ZAI_API_KEY': bool(os.environ.get('ZAI_API_KEY'))}
for k,v in keys.items(): print(f'  {\"✅\" if v else \"❌\"} {k}')
missing = [k for k,v in keys.items() if not v]
if missing: print(f'  ⚠️ CONSTITUTION RULE 3 VIOLATION: Missing model keys!')
" 2>/dev/null

echo "=== POLICY STATE ==="
cat artifacts/state/research_policy_state.json 2>/dev/null || echo "No policy state"
cat artifacts/state/research_policy_control.json 2>/dev/null || echo "No policy control"

echo "=== LATEST SIGNAL ==="
ls -lt "$PROJECT_ROOT/artifacts/live/signal_"*.json 2>/dev/null | head -1 || echo "No signals"
```

Report: one-line per section, flag issues. Vietnamese OK.

**Constitution violations to check:**
- Any project missing a terminal? → RULE 2 violation
- Model keys missing? → RULE 3 violation
- All terminals dead? → RULE 1 violation (NEVER STOP)
- Policy force-paused with no alternative work? → RULE 1 violation
