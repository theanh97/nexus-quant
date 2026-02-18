from __future__ import annotations

import json
from pathlib import Path

from .promote import promote_best_params


def promote_main(*, config_path: Path, best_params_path: Path, artifacts_dir: Path, apply: bool) -> int:
    out = promote_best_params(
        config_path=config_path,
        best_params_path=best_params_path,
        artifacts_dir=artifacts_dir,
        apply=apply,
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0

