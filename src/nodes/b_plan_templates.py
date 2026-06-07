"""Plan template selection for B candidate generation."""
from __future__ import annotations

from .b_candidate_policy import load_policy_config, policy_cache_key
from .b_utils import get_scene_template


def is_supported_plan_template(template: list[str]) -> bool:
    # The current optimizer/action_hints path supports one activity plus one restaurant.
    return template.count("activity") == 1 and template.count("restaurant") == 1


def get_plan_templates(scene_type: str) -> list[list[str]]:
    fallback = get_scene_template(scene_type)
    policy = load_policy_config(policy_cache_key())
    template_policy = policy.get("template_policy")
    if not isinstance(template_policy, dict):
        return [fallback]

    scene_templates = template_policy.get("scene_templates")
    raw_templates = []
    if isinstance(scene_templates, dict):
        raw_templates.append(scene_templates.get(scene_type))
    raw_templates.append(template_policy.get("default_template"))

    templates = []
    seen = set()
    for raw_template in raw_templates:
        if not isinstance(raw_template, list):
            continue
        template = [str(step).strip() for step in raw_template if str(step).strip()]
        key = tuple(template)
        if not is_supported_plan_template(template) or key in seen:
            continue
        seen.add(key)
        templates.append(template)

    return templates or [fallback]
