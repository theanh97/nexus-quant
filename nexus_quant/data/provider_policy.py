from __future__ import annotations

from typing import Any, Dict, List


REAL_DATA_PROVIDERS = frozenset({"binance_rest_v1", "local_csv_v1"})
SYNTHETIC_DATA_PROVIDERS = frozenset({"synthetic_perp_v1"})


def classify_provider(provider: str | None) -> str:
    name = str(provider or "").strip().lower()
    if name in REAL_DATA_PROVIDERS:
        return "real"
    if name in SYNTHETIC_DATA_PROVIDERS:
        return "synthetic"
    return "unknown"


def is_real_provider(provider: str | None) -> bool:
    return classify_provider(provider) == "real"


def validate_run_data_policy(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        cfg = {}
    data_cfg = cfg.get("data") or {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    data_policy = cfg.get("data_policy") or {}
    if not isinstance(data_policy, dict):
        data_policy = {}

    run_name = str(cfg.get("run_name") or "").strip()
    provider = str(data_cfg.get("provider") or "").strip()
    provider_class = classify_provider(provider)

    allowed_raw = data_policy.get("allowed_providers")
    allowed_providers: List[str] = []
    if isinstance(allowed_raw, list):
        allowed_providers = [str(x).strip() for x in allowed_raw if str(x).strip()]
    elif isinstance(allowed_raw, str):
        allowed_providers = [x.strip() for x in allowed_raw.split(",") if x.strip()]

    require_real_data = bool(data_policy.get("require_real_data") is True)
    if (not require_real_data) and run_name.lower().startswith("production_"):
        require_real_data = True

    allow_synthetic = bool(data_policy.get("allow_synthetic") is True)
    strict_unknown_provider = bool(data_policy.get("strict_unknown_provider") is True)

    errors: List[str] = []
    warnings: List[str] = []

    if not provider:
        errors.append("data.provider is required.")
    if allowed_providers and provider and provider not in set(allowed_providers):
        errors.append(f"Provider '{provider}' is not in allowed_providers={allowed_providers}.")
    if require_real_data and provider_class != "real":
        errors.append(
            f"Run requires real data but provider '{provider}' is '{provider_class}'."
        )
    if strict_unknown_provider and provider_class == "unknown":
        errors.append(f"Provider '{provider}' is unknown under strict_unknown_provider.")

    if provider_class == "synthetic" and not allow_synthetic:
        run_name_l = run_name.lower()
        looks_demo = run_name_l.startswith("synthetic_") or run_name_l.endswith("_demo") or ("demo" in run_name_l)
        if looks_demo:
            warnings.append(
                "Synthetic provider detected in demo run. Never use demo results for champion ranking or production."
            )
        else:
            errors.append(
                "Synthetic provider is blocked for non-demo runs unless data_policy.allow_synthetic=true."
            )

    return {
        "ok": not errors,
        "run_name": run_name,
        "provider": provider,
        "provider_class": provider_class,
        "is_real_data": provider_class == "real",
        "require_real_data": require_real_data,
        "allow_synthetic": allow_synthetic,
        "strict_unknown_provider": strict_unknown_provider,
        "allowed_providers": allowed_providers,
        "errors": errors,
        "warnings": warnings,
    }
