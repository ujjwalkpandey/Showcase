from agno import Agent, Context


class DataAgent(Agent):
    name = "data_agent"

    def run(self, ctx: Context):
        """
        Entry normalization agent
        """
        input_data = ctx.state.get("input")
        if not input_data:
            raise ValueError("DataAgent: input missing")

        if isinstance(input_data, dict):
            raw_text = "\n".join(f"{k}: {v}" for k, v in input_data.items())
        else:
            raw_text = str(input_data)

        ctx.state["raw_text"] = raw_text.strip()
