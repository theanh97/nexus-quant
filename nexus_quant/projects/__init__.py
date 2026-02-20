"""
NEXUS Projects — Multi-market project management.

Each project = 1 market/strategy domain with its own:
- providers (data sources)
- strategies (trading logic)
- costs (venue fee models)
- configs (run configurations)
- memory (project-specific wisdom at L2)

Projects share the NEXUS core engine, agents, brain, and orchestration.
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.projects")

# Global registry of loaded projects
_PROJECTS: Dict[str, "ProjectManifest"] = {}


class ProjectManifest:
    """Parsed project.yaml manifest."""

    def __init__(
        self,
        name: str,
        market: str,
        description: str = "",
        asset_class: str = "",
        universe: List[str] = None,
        default_config: str = "",
        data_provider: str = "",
        cost_model: str = "",
        features: List[str] = None,
        strategies: List[str] = None,
        brain_interval: int = 600,
        enabled: bool = True,
        project_dir: Optional[Path] = None,
    ):
        self.name = name
        self.market = market
        self.description = description
        self.asset_class = asset_class
        self.universe = universe or []
        self.default_config = default_config
        self.data_provider = data_provider
        self.cost_model = cost_model
        self.features = features or []
        self.strategies = strategies or []
        self.brain_interval = brain_interval
        self.enabled = enabled
        self.project_dir = project_dir

    @classmethod
    def from_file(cls, path: Path) -> "ProjectManifest":
        """Load from project.yaml or project.json."""
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            # Minimal YAML parser (stdlib-only, no PyYAML dependency)
            data = _parse_simple_yaml(text)
        else:
            data = json.loads(text)
        manifest = cls(
            name=str(data.get("name", path.parent.name)),
            market=str(data.get("market", "unknown")),
            description=str(data.get("description", "")),
            asset_class=str(data.get("asset_class", "")),
            universe=list(data.get("universe", [])),
            default_config=str(data.get("default_config", "")),
            data_provider=str(data.get("data_provider", "")),
            cost_model=str(data.get("cost_model", "")),
            features=list(data.get("features", [])),
            strategies=list(data.get("strategies", [])),
            brain_interval=int(data.get("brain_interval", 600)),
            enabled=bool(data.get("enabled", True)),
            project_dir=path.parent,
        )
        return manifest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "market": self.market,
            "description": self.description,
            "asset_class": self.asset_class,
            "universe": self.universe,
            "default_config": self.default_config,
            "data_provider": self.data_provider,
            "cost_model": self.cost_model,
            "features": self.features,
            "strategies": self.strategies,
            "brain_interval": self.brain_interval,
            "enabled": self.enabled,
            "project_dir": str(self.project_dir) if self.project_dir else None,
        }


def discover_projects(projects_dir: Optional[Path] = None) -> Dict[str, ProjectManifest]:
    """Scan projects/ directory and load all project manifests."""
    if projects_dir is None:
        projects_dir = Path(__file__).parent
    results: Dict[str, ProjectManifest] = {}
    if not projects_dir.is_dir():
        return results
    for child in sorted(projects_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        for fname in ("project.yaml", "project.yml", "project.json"):
            manifest_path = child / fname
            if manifest_path.exists():
                try:
                    m = ProjectManifest.from_file(manifest_path)
                    results[m.name] = m
                    logger.info("Discovered project: %s (%s)", m.name, m.market)
                except Exception as exc:
                    logger.warning("Failed to load project %s: %s", child.name, exc)
                break
    return results


def load_projects(projects_dir: Optional[Path] = None) -> Dict[str, ProjectManifest]:
    """Discover and cache all projects."""
    global _PROJECTS
    _PROJECTS = discover_projects(projects_dir)
    return _PROJECTS


def get_project(name: str) -> Optional[ProjectManifest]:
    """Get a loaded project by name."""
    if not _PROJECTS:
        load_projects()
    return _PROJECTS.get(name)


def list_projects() -> List[str]:
    """List all loaded project names."""
    if not _PROJECTS:
        load_projects()
    return list(_PROJECTS.keys())


def get_active_projects() -> Dict[str, ProjectManifest]:
    """Get all enabled projects."""
    if not _PROJECTS:
        load_projects()
    return {k: v for k, v in _PROJECTS.items() if v.enabled}


# ── Minimal YAML parser (stdlib-only) ────────────────────────────────────

def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Parse a flat/simple YAML file into a dict.
    Handles: scalars, simple lists (- item), but NOT nested objects.
    Good enough for project.yaml manifests.
    """
    result: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_list: Optional[List[Any]] = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item
        if stripped.startswith("- "):
            if current_key and current_list is not None:
                val = stripped[2:].strip().strip('"').strip("'")
                current_list.append(_yaml_cast(val))
            continue

        # Key: value
        if ":" in stripped:
            # Save previous list
            if current_key and current_list is not None:
                result[current_key] = current_list

            parts = stripped.split(":", 1)
            key = parts[0].strip()
            val_str = parts[1].strip() if len(parts) > 1 else ""

            if not val_str:
                # Start of a list
                current_key = key
                current_list = []
            else:
                current_key = None
                current_list = None
                result[key] = _yaml_cast(val_str.strip('"').strip("'"))

    # Save last list
    if current_key and current_list is not None:
        result[current_key] = current_list

    return result


def _yaml_cast(val: str) -> Any:
    """Cast a YAML string value to appropriate Python type."""
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "null" or val == "~":
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val
