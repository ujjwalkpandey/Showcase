import json
from pathlib import Path
from agno import Agent, Context

TEMPLATE_ROOT = Path("showcase/app/templates")
REGISTRY_PATH = TEMPLATE_ROOT / "registry.json"


class TemplateSelectorAgent(Agent):
    name = "template_selector_agent"

    def run(self, ctx: Context):
        profile = ctx.state.get("profile")
        if not profile:
            raise ValueError("TemplateSelectorAgent: profile missing")

        registry = self._load_registry()
        template_id = self._select_template(profile, registry)
        meta = registry[template_id]

        ctx.state["template"] = {
            "id": template_id,
            "name": meta["name"],
            "files": meta["files"],
            "path": str(TEMPLATE_ROOT / f"{template_id}_{meta['name'].lower()}")
        }

    def _load_registry(self):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _select_template(self, profile, registry):
        role = profile.get("role", "").lower()
        exp = profile.get("experience_years", 0)

        if "developer" in role and "t03" in registry:
            return "t03"
        if exp >= 5 and "t02" in registry:
            return "t02"
        return "t01"
