from agno import Application

from agents.core.data_agent import DataAgent
from agents.core.prompt_agent import PromptAgent
from agents.core.gemini_agent import GeminiAgent
from agents.core.schema_builder import SchemaBuilderAgent
from agents.core.template_selector_agent import TemplateSelectorAgent


def create_app():
    return Application(
        name="Showcase-Agno-App",
        agents=[
            DataAgent(),
            PromptAgent(),
            GeminiAgent(),
            SchemaBuilderAgent(),
            TemplateSelectorAgent(),
        ],
    )


app = create_app()
