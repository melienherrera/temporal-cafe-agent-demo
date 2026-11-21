import os
from datetime import timedelta
from typing import Annotated

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream

with workflow.unsafe.imports_passed_through():
    import httpx
    from agents import Agent as OpenAIAgent
    from agents import Runner, TResponseInputItem
    from pydantic import BaseModel, Field
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ApplicationError
    from temporalio.workflow import ActivityConfig

    from temporal_agent_kit.ai_sdks.openai_agents import as_openai_agent_tool
    from temporal_agent_kit import agent
    from temporal_agent_kit.agent_protocol import (
        AgentConfig,
        TextMessage,
        TextReply,
        ToolApprovalPolicy,
    )
    from temporal_agent_kit.agent_workflow import AgentWorkflowRunner


TASK_QUEUE = "barista"
DEFAULT_MODEL = "gpt-5.1"

# External services the agent's activity tools call. Both reads/writes become durable
# Temporal activities, so a crash or a flaky response is retried, not lost.
POS_URL = os.environ.get("POS_URL", "http://localhost:9000")
OFF_SEARCH_URL = "https://search.openfoodfacts.org/search"
HTTP_UA = "MelissaCafeBaristaAgent/0.1 (Durable AI demo)"

# Pricing, in cents. The agent quotes this before placing the order, and `place_order`
# charges exactly it — computed from stable inputs so a retry charges the same amount.
SIZE_PRICE_CENTS = {"small": 450, "medium": 525, "large": 600}
ALT_MILK_SURCHARGE_CENTS = 65
ALT_MILKS = {"oat", "almond", "cashew", "soy", "coconut"}

# Food menu: name -> (price_cents, ingredients). Pastries are priced here in Python, like drinks —
# the model never prices them. Ingredients are listed so the agent can screen for allergens and
# verify anything uncertain with check_ingredient before recommending an item.
FOOD_MENU: dict[str, tuple[int, str]] = {
    "butter croissant": (395, "butter, wheat flour, yeast, salt"),
    "pain au chocolat": (425, "butter, wheat flour, dark chocolate, yeast"),
    "almond croissant": (495, "butter, wheat flour, almond frangipane (almonds), powdered sugar"),
    "blueberry muffin": (375, "wheat flour, egg, butter, blueberries, sugar"),
    "cinnamon roll": (450, "wheat flour, butter, egg, cinnamon, cream cheese"),
    "lemon poppyseed loaf": (400, "wheat flour, egg, butter, lemon, poppy seeds"),
    "chocolate chip cookie": (325, "wheat flour, butter, egg, chocolate chips, brown sugar"),
    "everything bagel": (350, "wheat flour, yeast, sesame, poppy seeds, garlic, onion"),
    "fruit danish": (425, "wheat flour, butter, egg, seasonal fruit, custard"),
    "pecan sticky bun": (475, "wheat flour, butter, egg, pecans, caramel"),
    "pistachio financier": (450, "egg white, butter, pistachios, almond flour, sugar"),
    "baklava": (425, "phyllo pastry, walnuts, pistachios, honey, butter"),
    "banana walnut bread": (395, "wheat flour, egg, butter, banana, walnuts"),
    "vegan blueberry muffin": (395, "wheat flour, coconut oil, blueberries, oat milk, sugar"),
    "vegan chocolate chip cookie": (350, "wheat flour, coconut oil, dark chocolate, brown sugar"),
    "vegan oat & seed bar": (375, "oats, sunflower seeds, pumpkin seeds, maple syrup, coconut oil"),
    "vegan banana walnut muffin": (415, "wheat flour, banana, walnuts, coconut oil, oat milk"),
}


def _price_cents(size: str, milk: str) -> int:
    """Unit price of one drink, in cents, from its size and milk. Pure Python — the model
    never computes prices; it only assembles the cart."""
    base = SIZE_PRICE_CENTS.get(size.strip().lower(), SIZE_PRICE_CENTS["medium"])
    if milk.strip().lower() in ALT_MILKS:
        base += ALT_MILK_SURCHARGE_CENTS
    return base


class OrderItem(BaseModel):
    """One line in a café order. The model fills these in from the conversation; it does not
    price them."""

    drink: str = Field(..., description="The drink, e.g. 'oat latte' or 'cortado'.")
    milk: str = Field("whole", description="Milk to use, e.g. 'whole', 'oat', 'almond'.")
    size: str = Field("medium", description="One of: small, medium, large.")
    quantity: int = Field(1, ge=1, description="How many of this exact drink.")


class FoodItem(BaseModel):
    """One food line in an order — a pastry or snack. The model fills these from the conversation
    (names come from get_food); it does not price them."""

    name: str = Field(
        ..., description="The food item's name, exactly as listed in get_food, e.g. 'butter croissant'."
    )
    quantity: int = Field(1, ge=1, description="How many of this item.")


def _food_price_cents(name: str) -> int | None:
    """Unit price of one food item, in cents, or None if we don't carry it."""
    entry = FOOD_MENU.get(name.strip().lower())
    return entry[0] if entry else None


def _price_cart(
    items: list[OrderItem], food: list[FoodItem]
) -> tuple[list[tuple[str, int]], int, list[str]]:
    """Price a full cart (drinks + food), in cents — all arithmetic in Python, none in the model.

    Returns (lines, total_cents, unknown_food): each line is (description, line_total_cents) and a
    line total is unit price times quantity. Food names we don't carry are collected, not priced."""
    lines: list[tuple[str, int]] = []
    unknown: list[str] = []
    for it in items:
        lines.append((_describe_item(it), _price_cents(it.size, it.milk) * it.quantity))
    for f in food:
        unit = _food_price_cents(f.name)
        if unit is None:
            unknown.append(f.name)
            continue
        lines.append((_describe_food(f), unit * f.quantity))
    total_cents = sum(line_total for _, line_total in lines)
    return lines, total_cents, unknown


def _fmt_cents(cents: int) -> str:
    """Canonical money string: 1780 -> '$17.80'. One formatter so the quote the customer sees,
    the total on the approval card, and `place_order`'s re-check all agree to the character."""
    return f"${cents / 100:.2f}"


def _describe_item(it: OrderItem) -> str:
    qty = f"{it.quantity}x " if it.quantity != 1 else ""
    return f"{qty}{it.size} {it.drink} ({it.milk} milk)"


def _describe_food(f: FoodItem) -> str:
    qty = f"{f.quantity}x " if f.quantity != 1 else ""
    return f"{qty}{f.name}"

SYSTEM_INSTRUCTION = """\
You are a warm, friendly barista at Melissa's Café. Customers order in plain language and may
not know exactly what they want.

How you work:
- If the order is vague about something that matters (size, strength, hot vs iced, milk,
  sweetness), ask ONE short, friendly clarifying question before making anything. One good
  question at a time — don't interrogate.
- Stay honest with your tools: call `get_menu` for drinks and `get_food` for pastries/snacks to see
  what's on offer, and `check_milk` before promising a milk — don't guess what's in stock.
- If the customer asks about allergens or dietary fit (vegan, gluten, nut-free, "is oat milk
  dairy?"), call `check_ingredient` to look it up before answering. Don't guess dietary facts.
- Remember any allergy or diet a customer mentions for the rest of the chat. When they ask what
  food is safe, call `get_food`, read each item's ingredients, use `check_ingredient` to verify
  anything you're unsure about, and recommend ONLY items you've confirmed are safe for them.
- A customer may order drinks and/or food. First collect everything — drinks (drink, size, milk,
  quantity) and food (name, quantity). You never price anything yourself — `quote_order` is the only
  source of prices, for both answering "how much?" and totaling an order. To place an order, do two
  steps in order: (1) call `quote_order` with the drinks and food to get the exact total; (2) call
  `place_order` with those same drinks and food and `quoted_total` set to the exact total string
  `quote_order` returned. Then report the order number and total from the receipt. Placing an order
  takes payment, so it needs a quick approval.
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
async def get_food() -> str:
    """Return the café's food menu (pastries and snacks), each with its ingredients so you can
    screen for allergens. Prices come from quote_order, not here."""
    lines = "\n".join(
        f"{name.title()} — {ingredients}" for name, (_price, ingredients) in FOOD_MENU.items()
    )
    return (
        "FOOD MENU (ingredients listed so you can check allergens; ask quote_order for prices)\n"
        + lines
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


@agent.activity_tool_defn(
    inherently_safe=True,  # read-only external lookup; safe to run without approval
    activity_config=ActivityConfig(
        start_to_close_timeout=timedelta(seconds=15),
        retry_policy=RetryPolicy(
            initial_interval=timedelta(seconds=1), maximum_attempts=4
        ),
    ),
)
async def check_ingredient(item: str) -> str:
    """Look up allergen and dietary info for an ingredient or product via Open Food Facts.

    Runs as a durable Temporal activity: a real HTTP read to an external database, retried
    automatically if it fails. `item` is a free-text ingredient or product, e.g. "oat milk",
    "almond milk", "vanilla syrup". Returns the top match's allergens and vegan/vegetarian
    tags — honest data for answering a dietary question."""
    async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": HTTP_UA}) as client:
        resp = await client.get(
            OFF_SEARCH_URL,
            params={
                "q": item,
                "page_size": 1,
                "fields": "product_name,brands,allergens_tags,ingredients_analysis_tags",
            },
        )
        resp.raise_for_status()
        hits = resp.json().get("hits") or []

    if not hits:
        return f"No Open Food Facts match for {item!r} — I can't confirm its dietary info."

    top = hits[0]
    name = top.get("product_name") or item
    allergens = [t.split(":", 1)[-1] for t in (top.get("allergens_tags") or [])]
    diet = [t.split(":", 1)[-1] for t in (top.get("ingredients_analysis_tags") or [])]
    allergen_text = ", ".join(allergens) if allergens else "none listed"
    diet_text = ", ".join(diet) if diet else "no dietary tags"
    return f"Top match for {item!r}: {name}. Allergens: {allergen_text}. Tagged: {diet_text}."


@agent.tool_defn(inherently_safe=True)
async def quote_order(items: list[OrderItem], food: list[FoodItem]) -> str:
    """Price a cart BEFORE placing it, so you can tell the customer the total and hand the exact
    figure to `place_order`. Pure Python — the model never does the arithmetic, it just relays
    this. Returns the per-line breakdown and the exact total; pass that total to
    `place_order(quoted_total=...)` verbatim.

    `items` is the drinks (drink, milk, size, quantity); `food` is the pastries/snacks (name,
    quantity). Either list may be empty, but include everything the customer is ordering."""
    lines, total_cents, unknown = _price_cart(items, food)
    if unknown:
        return (
            f"I don't recognize these food items: {', '.join(unknown)}. "
            "Check get_food for the exact names."
        )
    if not lines:
        return "Nothing to quote yet — add at least one drink or food item."
    line_text = "\n".join(f"  • {desc} — {_fmt_cents(line_total)}" for desc, line_total in lines)
    total = _fmt_cents(total_cents)
    return f'{line_text}\n  Total: {total}\nPass quoted_total="{total}" to place_order.'


@agent.activity_tool_defn(
    inherently_safe=False,  # takes payment -> gated for human approval (HITL)
    activity_config=ActivityConfig(
        start_to_close_timeout=timedelta(seconds=15),
        retry_policy=RetryPolicy(
            initial_interval=timedelta(seconds=1), maximum_attempts=5
        ),
    ),
)
async def place_order(
    items: list[OrderItem],
    food: list[FoodItem],
    quoted_total: Annotated[
        str,
        Field(
            description="The exact total string returned by quote_order, e.g. '$17.80'. You MUST "
            "call quote_order first (with the drinks and food) and copy its total here verbatim — "
            "never compute it, guess, or pass 0. place_order rejects a total that doesn't match."
        ),
    ],
    idempotency_key: agent.Injected[str],
) -> str:
    """Charge for a whole order (drinks and/or food) and send it to the counter. Requires
    approval before it runs — the approver sees the items and `quoted_total` on the card.

    Runs as a durable Temporal activity against the café POS. The total is computed HERE in
    Python from `items` and `food` — never by the model — so it's always right and matches what
    the POS charges. `quoted_total` is the figure `quote_order` produced and the customer approved;
    this tool re-derives the total and refuses if they disagree, so the amount on the card is always
    the amount charged. `idempotency_key` is supplied by the workflow (the model never sets it): the
    POS dedupes on it, so a retry, or a resume after a worker crash mid-charge, returns the same
    order instead of charging twice.

    `items` is the drinks; `food` is the pastries/snacks. Either may be empty, but not both."""
    lines, total_cents, unknown = _price_cart(items, food)
    if unknown:
        raise ApplicationError(
            f"These food items aren't on the menu: {', '.join(unknown)}. Check get_food for the "
            "exact names, then re-quote and place the order.",
            non_retryable=True,
        )
    if not lines:
        raise ApplicationError(
            "Nothing to place — the order has no drinks or food.", non_retryable=True
        )
    authoritative_total = _fmt_cents(total_cents)
    if quoted_total.strip() != authoritative_total:
        raise ApplicationError(
            f"quoted_total {quoted_total!r} doesn't match the register total "
            f"{authoritative_total} for this order — call quote_order again and place the "
            f"order with the exact total it returns.",
            non_retryable=True,
        )
    description = "; ".join(desc for desc, _ in lines)

    milks = {it.milk for it in items}
    sizes = {it.size for it in items}
    async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": HTTP_UA}) as client:
        resp = await client.post(
            f"{POS_URL}/orders",
            json={
                "idempotency_key": idempotency_key,
                "drink": description,
                "milk": "mixed" if len(milks) > 1 else (next(iter(milks)) if milks else "n/a"),
                "size": "mixed" if len(sizes) > 1 else (next(iter(sizes)) if sizes else "n/a"),
                "amount_cents": total_cents,
            },
        )
        resp.raise_for_status()
        receipt = resp.json()

    line_text = "\n".join(f"  • {desc} — {_fmt_cents(line_total)}" for desc, line_total in lines)
    total = receipt["amount_cents"] / 100
    note = " (already paid — not charged again)" if receipt.get("deduped") else ""
    return (
        f"☕ Order {receipt['order_id']} is paid{note}:\n{line_text}\n"
        f"  Total: ${total:.2f}. Sent to the counter — enjoy!"
    )


# The agent's full toolset. Pass this to AgentKitPlugin(tools=...) on the worker: it
# registers the durable activity bodies of the activity tools (check_ingredient, place_order)
# and skips the inline ones (get_menu, get_food, check_milk, quote_order), which have no worker-side body.
TOOLS = [get_menu, get_food, check_milk, quote_order, check_ingredient, place_order]


@agent.defn(name="BaristaAgent")
class BaristaAgentWorkflow:
    """A café barista driven by the OpenAI Agents SDK: takes natural-language orders, checks the
    menu, milk, and dietary info, then charges and places the order after a human approval."""

    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            # Safe tools (get_menu, check_milk, check_ingredient) run directly; place_order
            # is not marked safe, so it pauses for approval in the console before it charges.
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
        # One stable, replay-safe idempotency key per order turn. workflow.uuid4() is
        # deterministic under replay, so the SAME key reaches place_order on every retry —
        # that's what makes the POS charge exactly-once. The model never sees it (Injected).
        idempotency_key = str(workflow.uuid4())

        sdk_agent = OpenAIAgent(
            name="Barista",
            instructions=SYSTEM_INSTRUCTION,
            model=DEFAULT_MODEL,
            tools=[
                as_openai_agent_tool(self._runner, get_menu),
                as_openai_agent_tool(self._runner, get_food),
                as_openai_agent_tool(self._runner, check_milk),
                as_openai_agent_tool(self._runner, check_ingredient),
                as_openai_agent_tool(self._runner, quote_order),
                as_openai_agent_tool(
                    self._runner,
                    place_order,
                    injections={"idempotency_key": idempotency_key},
                ),
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
