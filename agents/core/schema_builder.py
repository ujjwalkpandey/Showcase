"""
SCHEMA_BUILDER.PY - Portfolio Schema Construction Agent

PURPOSE:
This agent transforms raw parsed resume data into a structured portfolio schema.
It analyzes the user's profile and creates a blueprint for content generation.

DATA FLOW IN:
- Preprocessed resume data (name, skills, projects, experience, education)

DATA FLOW OUT:
- Structured schema with:
  * Hero line template
  * Bio structure
  * Project entries with categorization
  * Skill groupings
  * Layout hints (what to emphasize)

HOW IT WORKS:
- Analyzes user's domain (developer, designer, data scientist, etc.)
- Determines best portfolio structure for their profile
- Creates templates that the content generator will fill
- Suggests visual hierarchy and emphasis points

NOTE: THIS CODE IS AI GENERATED, YOUR WORK IS TO ANALYSIS THE CODE AND CHECK THE LOGIC AND MAKE CHANGES
     WHERE REQUIRED
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class SchemaBuilder:
    """
    Builds a neutral, data-driven portfolio schema from preprocessed input.

    This version intentionally avoids:
    - Role or domain assumptions
    - Fixed skill categories
    - Prescriptive visual or content styles
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("Generic SchemaBuilder initialized")

    async def build_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct a generalized schema that downstream systems
        (LLMs, renderers, agents) can interpret freely.
        """
        try:
            logger.info("Building generic portfolio schema")

            schema = {
                "profile_summary": self._build_profile_summary(data),
                "hero": self._build_hero_schema(data),
                "bio": self._build_bio_schema(data),
                "projects": self._build_projects_schema(data),
                "skills": self._build_skills_schema(data),
                "layout_hints": self._build_layout_hints(data),
                "generation_flags": {
                    "hero": True,
                    "bio": True,
                    "projects": True
                }
            }

            return schema

        except Exception as e:
            logger.exception("Schema construction failed")
            raise e

    def _build_profile_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract high-level signals without classification.
        """
        return {
            "name": data.get("name"),
            "has_projects": bool(data.get("projects")),
            "has_experience": bool(data.get("experience")),
            "has_education": bool(data.get("education")),
            "skill_count": len(data.get("skills", []))
        }

    def _build_hero_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Neutral hero section. No role-based language.
        """
        return {
            "name": data.get("name", "Portfolio"),
            "tagline_placeholder": "Concise professional summary",
            "contact": {
                "email": data.get("email"),
                "links": data.get("links", {})
            }
        }

    def _build_bio_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structural guidance only.
        """
        return {
            "sections": [
                "introduction",
                "background",
                "work_highlights",
                "current_interests",
                "closing"
            ],
            "reference_points": self._extract_bio_points(data),
            "length_hint": "medium"
        }

    def _extract_bio_points(self, data: Dict[str, Any]) -> List[str]:
        points = []

        if data.get("education"):
            edu = data["education"][0]
            points.append(
                f"Education: {edu.get('degree', '')} {edu.get('institution', '')}".strip()
            )

        if data.get("experience"):
            points.append(f"Experience entries: {len(data['experience'])}")

        if data.get("projects"):
            points.append(f"Projects included: {len(data['projects'])}")

        if data.get("skills"):
            points.append(f"Skills listed: {', '.join(data['skills'][:5])}")

        return points

    def _build_projects_schema(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        projects = data.get("projects", [])
        result = []

        for idx, project in enumerate(projects):
            result.append({
                "id": f"project_{idx}",
                "title": project.get("title", f"Project {idx + 1}"),
                "description_source": project.get("description", ""),
                "technologies": project.get("technologies", []),
                "links": project.get("links", {}),
                "priority": "high" if idx < 3 else "normal"
            })

        return result

    def _build_skills_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        No categorization assumptions.
        """
        skills = data.get("skills", [])

        return {
            "raw": skills,
            "count": len(skills),
            "grouping": "deferred"  # left to downstream logic
        }

    def _build_layout_hints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lightweight hints only.
        """
        hints = {
            "sections": ["hero", "projects", "skills"],
            "optional": [],
            "density": "balanced"
        }

        if len(data.get("experience", [])) > 1:
            hints["sections"].insert(2, "experience")
        else:
            hints["optional"].append("experience")

        return hints
