from __future__ import annotations

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream

with workflow.unsafe.imports_passed_through():
    from agents import Agent as OpenAIAgent
    from agents import Runner, TResponseInputItem

    from temporal_agent_harness.ai_sdks.openai_agents_harness import as_openai_agent_tool
    from temporal_agent_harness.harness import agent
    from temporal_agent_harness.harness.agent_protocol import (
        AgentConfig,
        TextMessage,
        TextReply,
        ToolApprovalPolicy,
    )
    from temporal_agent_harness.harness.agent_workflow import AgentWorkflowRunner


TASK_QUEUE = "barista"
DEFAULT_MODEL = "gpt-5.1"

SYSTEM_INSTRUCTION = """\
You are a warm, friendly barista at Melissa's Café. Customers order in plain language and may
not know exactly what they want.

How you work:
- If the order is vague about something that matters (size, strength, hot vs iced, milk,
  sweetness), ask ONE short, friendly clarifying question before making anything. One good
  question at a time — don't interrogate.
- Stay honest with your tools: call `get_menu` to see what's on offer, and `check_milk` before
  promising a milk — don't guess what's in stock.
- When you have enough to make a great drink, call `make_drink` to prepare it. Making a drink
  needs a quick barista approval; that's expected.
- Keep replies warm and brief, like a real counter chat. If the customer is unsure, suggest
  something you'd recommend."""


@agent.tool_defn(inherently_safe=True)
async def get_menu() -> str:
    """Return the café's current drink menu with short descriptions."""
    return (
        "ESPRESSO BAR\n"
        "Espresso — a classic short shot, served as a double.\n"
        "Macchiato — espresso marked with a spoon of foam.\n"
        "Cortado — espresso cut with a little warm milk.\n"
        "Gibraltar — a slightly longer cortado in a glass, silky and balanced.\n"
        "Cappuccino — espresso with equal steamed milk and foam.\n"
        "Flat white — espresso + silky microfoam, stronger than a latte.\n"
        "Latte — espresso + steamed milk, smooth and mellow.\n"
        "Americano — espresso lengthened with hot water.\n"
        "Mocha — espresso, steamed milk, and dark chocolate.\n"
        "White mocha — espresso, steamed milk, and white chocolate.\n"
        "\n"
        "SIGNATURE LATTES\n"
        "Brown sugar latte — espresso, brown sugar, and a pinch of cinnamon.\n"
        "Vanilla bean latte — espresso with real vanilla, not syrup-sweet.\n"
        "Honey lavender latte — espresso, wildflower honey, and a whisper of lavender.\n"
        "Cardamom latte — espresso warmed with green cardamom.\n"
        "Rose latte — espresso with a light rose syrup, floral not perfumey.\n"
        "Pistachio latte — espresso and toasted pistachio, lightly sweet.\n"
        "Salted caramel latte — espresso, caramel, and a flake of sea salt.\n"
        "Maple oat latte — espresso, maple, and oat milk even if you pick another milk.\n"
        "Orange blossom latte — espresso with orange blossom and a little honey.\n"
        "\n"
        "BREWED COFFEE\n"
        "Batch brew — today's single-origin drip, rotating, served black or with cream.\n"
        "Pour-over — a slower cup of the same coffee, clearer and brighter.\n"
        "Mint cream iced coffee — brewed coffee, fresh mint, cream, and sweetness to taste.\n"
        "Velvet medium — medium roast, creamy, a little sweet; our everyday cup.\n"
        "Dark cocoa coffee — dark roast brewed coffee with cocoa and cream.\n"
        "Bright & light — a lighter roast, splash of cream, low sweetness.\n"
        "New Orleans — coffee with chicory, a little sweetened condensed milk.\n"
        "Hazelnut cream coffee — brewed coffee, hazelnut, and cream.\n"
        "\n"
        "COLD BAR\n"
        "Cold brew — slow-steeped 18 hours, smooth and low-acid (iced).\n"
        "Nitro cold brew — cold brew on tap, creamy with no milk.\n"
        "Sweet cream cold brew — cold brew topped with vanilla sweet cream.\n"
        "Shaken espresso — espresso shaken with ice until frothy.\n"
        "Brown sugar shaken espresso — shaken espresso, brown sugar, oat milk.\n"
        "Espresso tonic — espresso over tonic and ice, bright and bitter.\n"
        "Iced americano — espresso and cold water over ice.\n"
        "Iced latte — the latte, over ice.\n"
        "\n"
        "MATCHA & HOJICHA\n"
        "Ceremonial matcha — stone-ground matcha whisked with water, unsweetened.\n"
        "Matcha latte — ceremonial matcha and steamed milk, lightly sweet.\n"
        "Strawberry matcha — matcha latte with strawberry, best iced.\n"
        "Yuzu matcha — matcha with yuzu citrus, tart and bright.\n"
        "Brown sugar matcha — matcha, brown sugar, and milk.\n"
        "Coconut cloud matcha — matcha with coconut milk and a little vanilla.\n"
        "Vanilla matcha — matcha latte with vanilla.\n"
        "Ube matcha — matcha and purple yam, lightly sweet.\n"
        "Dirty matcha — matcha latte with a shot of espresso.\n"
        "Matcha lemonade — matcha over lemonade and ice, no milk.\n"
        "Hojicha latte — roasted green tea and steamed milk, toasty and low-caffeine.\n"
        "Iced hojicha — hojicha latte over ice.\n"
        "\n"
        "TEA & NON-COFFEE\n"
        "Chai latte — spiced black tea and steamed milk.\n"
        "Dirty chai — chai latte with a shot of espresso.\n"
        "London fog — Earl Grey, vanilla, and steamed milk.\n"
        "Golden milk — turmeric, ginger, honey, and steamed milk. No caffeine.\n"
        "Hot chocolate — dark chocolate and steamed milk.\n"
        "Steamer — steamed milk and a syrup (vanilla, brown sugar, lavender, rose). No caffeine."
    )


@agent.tool_defn(inherently_safe=True)
async def check_milk(kind: str) -> str:
    """Check whether a milk is in stock. `kind` is a milk name, e.g. "oat", "cashew", "whole"."""
    stock = {
        "whole": "in stock",
        "skim": "in stock",
        "oat": "in stock, barista-grade",
        "almond": "in stock",
        "cashew": "in stock",
        "soy": "in stock",
        "coconut": "out today, sorry",
    }
    status = stock.get(kind.strip().lower())
    if status is None:
        return f"We don't carry {kind}. We have whole, skim, oat, almond, cashew, and soy."
    return f"{kind.strip().title()} milk is {status}."


@agent.tool_defn(inherently_safe=False)
async def make_drink(drink: str, milk: str = "whole", iced: bool = False, notes: str = "") -> str:
    """Prepare and serve a drink. Requires barista approval before it runs.

    `drink` is the drink to make, e.g. "oat latte"; `milk` the milk to use; `iced` true for an
    iced drink; `notes` any extras like "extra shot" or "light sweetness"."""
    temp = "iced" if iced else "hot"
    extra = f" ({notes})" if notes else ""
    return f"☕ Order up: a {temp} {drink} with {milk} milk{extra}. Pulled a clean double shot — enjoy!"


@agent.defn(name="BaristaAgent")
class BaristaAgentWorkflow:
    """A café barista driven by the OpenAI Agents SDK: takes natural-language orders, checks the
    menu and milk, and makes the drink after a human approval."""

    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            # Safe tools (get_menu, check_milk) run directly; make_drink is not marked safe,
            # so it pauses for approval in the console before the drink is made.
            approval_policy_default=ToolApprovalPolicy.allow_inherently_safe(),
        )
        self._conversation: list[TResponseInputItem] = []

    @agent.accepts
    async def ping(self, message: TextMessage) -> TextReply:
        """Check the worker connection without calling a model."""
        return TextReply(text=message.text)

    @agent.accepts
    async def order(self, message: TextMessage) -> TextReply:
        """Order a coffee in plain language. The barista may ask a quick question, checks the
        menu and milk, then makes your drink (with a quick approval)."""
        sdk_agent = OpenAIAgent(
            name="Barista",
            instructions=SYSTEM_INSTRUCTION,
            model=DEFAULT_MODEL,
            tools=[
                as_openai_agent_tool(self._runner, get_menu),
                as_openai_agent_tool(self._runner, check_milk),
                as_openai_agent_tool(self._runner, make_drink),
            ],
        )
        input_items: list[TResponseInputItem] = [
            *self._conversation,
            {"role": "user", "content": message.text},
        ]

        result = Runner.run_streamed(sdk_agent, input=input_items, context=self._runner)
        async for _event in result.stream_events():
            pass

        self._conversation = result.to_input_list()
        return TextReply(text=str(result.final_output))
