"""
Custom Prophet Arena Trading Agent v2
Improvements:
1. Smart filtering: remove extreme-price markets and rank by spread to find the most uncertain ones
2. LLM-generated search queries: no longer use the raw question as the query
3. p_yes shrinkage: pull the LLM estimate back toward the market price to reduce overconfidence
4. Error recovery: OpenAI calls include built-in retries
5. Cross-tick memory: feed the previous tick's decisions and outcomes back to the LLM
"""

import json
import logging
import os
import time
import uuid

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from ai_prophet_core.client import ServerAPIClient, TradeIntentRequest

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

PA_API_URL  = "https://ai-prophet-core-api-998105805337.us-central1.run.app"
PA_API_KEY  = os.environ["PA_SERVER_API_KEY"]
OPENAI_KEY  = os.environ["OPENAI_API_KEY"]
BRAVE_KEY   = os.environ.get("BRAVE_API_KEY", "")

EXPERIMENT_SLUG       = "news_kelly_t1"
N_TICKS               = 1
STARTING_CASH         = 10_000.0
TOP_N_MARKETS         = 10
MAX_PER_MARKET        = 1_000.0
MIN_EDGE              = 0.01 # changed
MIN_BET               = 20.0
KELLY_FRACTION        = 0.25
SHRINKAGE             = 0   # [New] p_yes shrinkage factor: 0 = trust LLM fully, 1 = trust market fully
LOG_FILE              = "tick_log.jsonl"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def write_tick_log(record: dict) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    logger.info(f"Tick log written → {LOG_FILE}")


openai_client = OpenAI(api_key=OPENAI_KEY)


# ── [Improvement 4] Retried OpenAI Calls ─────────────────────────────────────

# ── [Improvement 4] Retried OpenAI Calls ─────────────────────────────────────

def call_openai(prompt: str, max_retries: int = 3, **kwargs) -> dict:
    """Call OpenAI with exponential-backoff retries and return parsed JSON."""

    # Default to 200 max tokens if the caller does not specify it
    kwargs.setdefault("max_tokens", 200)

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                **kwargs,  # Pass through any additional parameters, including max_tokens
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + 0.5
                logger.warning(f"OpenAI call failed (attempt {attempt+1}): {e}, retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise

# ── News Search ───────────────────────────────────────────────────────────────

def search_news(query: str, num_results: int = 5) -> str:
    """Search recent news with Brave Search and return summary text."""
    if not BRAVE_KEY:
        return ""
    query = query[:200]
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/news/search",
            params={"q": query, "count": num_results, "freshness": "pw"},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_KEY,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Brave API returned {resp.status_code} for '{query}'")
            return ""
        results = resp.json().get("results", [])
        lines = []
        for r in results:
            title = r.get("title", "")
            desc  = r.get("description", "")
            age   = r.get("age", "")
            lines.append(f"- [{age}] {title}: {desc}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Search error for '{query}': {e}")
        return ""


# ── [Improvement 2] LLM Search Query Generation ───────────────────────────────

def generate_search_queries(question: str) -> list[str]:
    """Use the LLM to turn a market question into 2-3 focused search queries."""
    prompt = f"""Convert this prediction market question into 2-3 short web search queries
that would find the most relevant recent news.

Question: {question}

Respond with ONLY a JSON object:
{{"queries": ["query1", "query2"]}}"""

    try:
        result = call_openai(prompt, max_tokens=100)
        queries = result.get("queries", [])
        # Fallback: if the LLM returns nothing or malformed output, use the raw question
        if not queries or not isinstance(queries, list):
            return [question[:200]]
        return [q[:200] for q in queries[:3]]
    except Exception as e:
        logger.warning(f"Query generation failed: {e}, falling back to raw question")
        return [question[:200]]


# ── LLM Forecasting ───────────────────────────────────────────────────────────

def predict_market(market, memory_text: str = "") -> tuple[float, str, str]:
    """
    Forecast p_yes for a market with GPT-4o.
    [Improvement 3] The returned p_yes is shrunk toward the market price.
    [Improvement 5] Accepts cross-tick memory text.
    """
    question   = market.question
    yes_ask    = float(market.quote.best_ask)
    no_ask     = 1.0 - float(market.quote.best_bid)
    mid_price  = (yes_ask + (1.0 - no_ask)) / 2  # Market mid-price
    resolution = market.resolution_time.strftime("%Y-%m-%d")

    # [Improvement 2] Search with LLM-generated queries instead of the raw question
    queries = generate_search_queries(question)
    all_news = []
    for q in queries:
        result = search_news(q)
        if result:
            all_news.append(f"[Query: {q}]\n{result}")
    news_block = "\nRecent news:\n" + "\n\n".join(all_news) if all_news else "\n(No recent news found)"

    # [Improvement 5] Cross-tick memory
    memory_block = f"\n\nYour notes from previous ticks:\n{memory_text}" if memory_text else ""

    prompt = f"""You are a sharp, independent contrarian analyst for prediction markets.

Market question: {question}
Resolves: {resolution}
Current market price (Implied Probability): {mid_price:.1%}
{news_block}{memory_block}

Your task: Determine if the market price is WRONG based on the latest news.

Rules:
- Many low-volume markets are outdated. If news shows a clear trend the price hasn't caught, be bold.
- If the news confirms the price, stay close to it.
- If you find a massive discrepancy (e.g., market says 50% but news says it's a done deal), explain why in the rationale.

Respond with ONLY a JSON object:
{{"p_yes": <float 0-1>, "rationale": "<why the market is right or wrong>"}}"""

    raw = call_openai(prompt, max_tokens=400)
    raw_p_yes = max(0.01, min(0.99, float(raw["p_yes"])))
    rationale = raw.get("rationale", "")

    # [Improvement 3] Shrink p_yes toward the market price
    # adjusted = (1 - SHRINKAGE) * LLM_estimate + SHRINKAGE * market_price
    adjusted_p_yes = (1.0 - SHRINKAGE) * raw_p_yes + SHRINKAGE * mid_price

    logger.info(
        f"    p_yes: raw={raw_p_yes:.1%} → shrunk={adjusted_p_yes:.1%} "
        f"(market mid={mid_price:.1%}, shrinkage={SHRINKAGE})"
    )

    return adjusted_p_yes, rationale, "\n\n".join(all_news)


# ── Position Sizing ───────────────────────────────────────────────────────────

def kelly_size(p_yes: float, yes_ask: float, no_ask: float, cash: float) -> tuple[str, float]:
    """Kelly Criterion position sizing (unchanged)."""
    yes_edge = p_yes - yes_ask
    no_edge  = (1.0 - p_yes) - no_ask

    if yes_edge >= no_edge and yes_edge > 0:
        side, edge, price = "YES", yes_edge, yes_ask
        p = p_yes
    elif no_edge > yes_edge and no_edge > 0:
        side, edge, price = "NO", no_edge, no_ask
        p = 1.0 - p_yes
    else:
        return "YES", 0.0

    if price <= 0 or price >= 1:
        return side, 0.0

    b = (1.0 / price) - 1.0
    kelly_f = max(0.0, (p * b - (1.0 - p)) / b)

    amount = min(cash * kelly_f * KELLY_FRACTION, MAX_PER_MARKET)
    return side, amount


# ── [Improvement 1] Smart Market Selection ────────────────────────────────────

def load_traded_market_ids() -> set[str]:
    """Load market IDs already traded from past tick logs to avoid repeat limits."""
    traded = set()
    if not os.path.exists(LOG_FILE):
        return traded
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for t in record.get("trades", []):
                    mid = t.get("market_id")
                    if mid:
                        traded.add(mid)
    except Exception as e:
        logger.warning(f"Could not load tick log: {e}")
    return traded


def select_top_markets(markets: list, n: int = TOP_N_MARKETS) -> list:
    """
    Hunting-mode filter:
    1. Exclude markets already near 0 or 1
    2. Exclude markets already traded to avoid rejection from limits
    3. Sort by 24-hour volume in ascending order
    4. Prioritize markets with zero or very low trading volume
    """
    already_traded = load_traded_market_ids()
    if already_traded:
        logger.info(f"Excluding {len(already_traded)} already-traded markets")

    filtered = []
    for m in markets:
        yes_ask = float(m.quote.best_ask)
        if 0.02 <= yes_ask <= 0.98 and m.market_id not in already_traded:
            filtered.append(m)

    # Sort by ascending volume: smaller volume comes first
    filtered.sort(key=lambda m: float(m.quote.volume_24h or 0))

    logger.info(f"Hunting Mode: {len(filtered)} markets found. Top {n} quietest markets selected.")
    return filtered[:n]


# ── [Improvement 5] Cross-Tick Memory Management ──────────────────────────────

def build_memory_text(prev_record: dict | None) -> str:
    """Compress key information from the previous tick into prompt-ready text."""
    if not prev_record:
        return ""

    lines = [
        f"Previous tick: {prev_record.get('tick_id', '?')}",
        f"Portfolio: cash=${prev_record.get('cash', 0):,.0f}, pnl=${prev_record.get('pnl', 0):+,.0f}",
    ]

    analyzed = prev_record.get("analyzed", [])
    traded = [a for a in analyzed if a.get("decision", "").startswith("TRADE")]
    skipped = [a for a in analyzed if a.get("decision", "").startswith("SKIP")]

    if traded:
        lines.append("Trades made:")
        for t in traded:
            lines.append(
                f"  - {t['market_id']}: {t['side']} ${t.get('amount', 0):.0f} "
                f"(p_yes={t.get('p_yes', 0):.1%}, edge={t.get('edge', 0):+.1%})"
            )

    if skipped:
        lines.append(f"Skipped {len(skipped)} markets (insufficient edge)")

    accepted = prev_record.get("accepted", 0)
    rejected = prev_record.get("rejected", 0)
    if accepted or rejected:
        lines.append(f"Execution: {accepted} accepted, {rejected} rejected")

    return "\n".join(lines)


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run():
    api = ServerAPIClient(base_url=PA_API_URL, api_key=PA_API_KEY)

    exp = api.create_or_get_experiment(
        slug=EXPERIMENT_SLUG,
        config_hash="news-kelly-t1-aggressive",
        config_json={
            "strategy": "news_search_kelly_t1",
            "model": "gpt-4o",
            "improvements": [
                "smart_filtering",
                "llm_search_queries",
                "p_yes_shrinkage",
                "retry_logic",
                "cross_tick_memory",
            ],
            "shrinkage": SHRINKAGE,
        },
        n_ticks=N_TICKS,
    )
    experiment_id = exp.experiment_id
    logger.info(f"Experiment: {experiment_id} | slug={EXPERIMENT_SLUG} | new={exp.created}")

    part = api.upsert_participant(
        experiment_id,
        model="custom:news-gpt4o-kelly-v3",
        rep=0,
        starting_cash=STARTING_CASH,
    )
    participant_idx = part.participant_idx
    logger.info(f"Participant idx: {participant_idx}")

    lease_owner = str(uuid.uuid4())
    tick_count  = 0
    prev_tick_record: dict | None = None  # [Improvement 5] Previous tick record

    while True:
        claim = api.claim_tick(experiment_id, lease_owner)

        if claim.no_tick_available:
            if claim.reason == "experiment_completed":
                logger.info(f"Experiment complete after {tick_count} ticks.")
                break
            wait = claim.retry_after_sec or 15
            logger.info(f"No tick yet ({claim.reason}), waiting {wait}s...")
            time.sleep(wait)
            continue

        tick_id     = claim.tick_id
        snapshot_id = claim.snapshot_id
        tick_count += 1
        logger.info(f"\n{'='*60}\nTick {tick_count} | {tick_id}\n{'='*60}")

        try:
            cash = STARTING_CASH
            portfolio = api.get_portfolio(experiment_id, participant_idx)
            if portfolio:
                cash = float(portfolio.cash)
                pnl  = float(portfolio.total_pnl)
                logger.info(
                    f"Portfolio: cash=${cash:,.2f} | equity=${float(portfolio.equity):,.2f} | pnl=${pnl:+,.2f}"
                )

            candidates  = api.get_candidates(claim.tick_ts, snapshot_id)
            top_markets = select_top_markets(candidates.markets)  # [Improvement 1]

            # [Improvement 5] Build cross-tick memory
            memory_text = build_memory_text(prev_tick_record)
            if memory_text:
                logger.info(f"Memory from previous tick: {len(memory_text)} chars")

            tick_record: dict = {
                "tick_id":      tick_id,
                "tick_count":   tick_count,
                "cash":         cash,
                "equity":       float(portfolio.equity) if portfolio else cash,
                "pnl":          float(portfolio.total_pnl) if portfolio else 0.0,
                "all_markets": [
                    {
                        "market_id": m.market_id,
                        "question":  m.question,
                        "yes_ask":   float(m.quote.best_ask),
                        "yes_bid":   float(m.quote.best_bid),
                        "vol24h":    float(m.quote.volume_24h or 0),
                    }
                    for m in candidates.markets
                ],
                "analyzed":  [],
                "trades":    [],
                "accepted":  0,
                "rejected":  0,
            }

            intents = []

            for i, market in enumerate(top_markets):
                market_id = market.market_id
                yes_ask   = float(market.quote.best_ask)
                no_ask    = 1.0 - float(market.quote.best_bid)

                logger.info(
                    f"  [{i+1}/{len(top_markets)}] {market.question[:70]}"
                    f"\n    Prices: YES={yes_ask:.1%}  NO={no_ask:.1%}  vol24h=${market.quote.volume_24h:,.0f}"
                )

                market_log: dict = {
                    "market_id": market_id,
                    "question":  market.question,
                    "yes_ask":   yes_ask,
                    "no_ask":    no_ask,
                    "vol24h":    float(market.quote.volume_24h or 0),
                    "resolution": market.resolution_time.isoformat(),
                    "p_yes":     None,
                    "rationale": None,
                    "edge":      None,
                    "side":      None,
                    "amount":    None,
                    "decision":  None,
                }

                try:
                    # [Improvements 2+3+4+5] Search-query generation, shrinkage, retries, and memory
                    p_yes, rationale, news_found = predict_market(market, memory_text=memory_text)
                    market_log["news"] = news_found
                    side, amount     = kelly_size(p_yes, yes_ask, no_ask, cash)

                    yes_edge = p_yes - yes_ask
                    no_edge  = (1.0 - p_yes) - no_ask
                    edge     = yes_edge if side == "YES" else no_edge

                    market_log.update({
                        "p_yes":     round(p_yes, 4),
                        "rationale": rationale,
                        "edge":      round(edge, 4),
                        "side":      side,
                        "amount":    round(amount, 2),
                    })

                    logger.info(
                        f"    Forecast: p_yes={p_yes:.1%} | edge={edge:+.1%} | "
                        f"size=${amount:.0f} | {rationale}"
                    )

                    if edge < MIN_EDGE:
                        market_log["decision"] = f"SKIP: edge {edge:.1%} < min {MIN_EDGE:.1%}"
                        logger.info(f"    → SKIP (edge {edge:.1%} < min {MIN_EDGE:.1%})")
                        tick_record["analyzed"].append(market_log)
                        continue
                    if amount < MIN_BET:
                        market_log["decision"] = f"SKIP: size ${amount:.0f} < min ${MIN_BET}"
                        logger.info(f"    → SKIP (size ${amount:.0f} < min ${MIN_BET})")
                        tick_record["analyzed"].append(market_log)
                        continue

                    market_log["decision"] = f"TRADE: BUY {side} ${amount:.0f}"
                    logger.info(f"    → TRADE: BUY {side} ${amount:.0f}")

                    intents.append(TradeIntentRequest(
                        market_id=market_id,
                        action="BUY",
                        side=side,
                        shares=str(int(amount)),
                        idempotency_key=f"{experiment_id}:{participant_idx}:{tick_id}:{i}",
                    ))

                    if len(intents) >= 10:
                        tick_record["analyzed"].append(market_log)
                        break

                except Exception as e:
                    market_log["decision"] = f"ERROR: {e}"
                    logger.warning(f"    Prediction failed for {market_id}: {e}")

                tick_record["analyzed"].append(market_log)

            # Submit trades
            if intents:
                result = api.submit_trade_intents(
                    experiment_id, participant_idx, tick_id,
                    candidates.candidate_set_id, intents,
                )
                tick_record["accepted"] = result.accepted
                tick_record["rejected"] = result.rejected
                tick_record["trades"]   = [
                    {"market_id": r.market_id, "action": r.action,
                     "side": r.side, "shares": r.shares}
                    for r in result.fills
                ]
                logger.info(
                    f"Submitted {len(intents)} intents → "
                    f"{result.accepted} accepted, {result.rejected} rejected"
                )
                for r in result.rejections:
                    logger.warning(f"  Rejected: {r.reason}")
            else:
                logger.info("No trades this tick (no edge found)")

            write_tick_log(tick_record)

            # [Improvement 5] Save the current tick record for the next tick
            prev_tick_record = tick_record

            api.finalize_participant(
                experiment_id, participant_idx, tick_id, status="COMPLETED"
            )
            api.complete_tick(experiment_id, tick_id)
            logger.info(f"Tick {tick_id} ✓")

        except Exception as e:
            logger.error(f"Tick failed: {e}", exc_info=True)
            try:
                api.finalize_participant(
                    experiment_id, participant_idx, tick_id,
                    status="FAILED", error_detail=str(e)[:500],
                )
                api.complete_tick(experiment_id, tick_id)
            except Exception:
                pass

    api.close()
    logger.info("Agent stopped.")


if __name__ == "__main__":
    logger.info("Starting News-Driven Kelly Agent t1")
    logger.info(f"Brave Search: {'ENABLED' if BRAVE_KEY else 'DISABLED (no key)'}")
    logger.info(f"Shrinkage: {SHRINKAGE}")
    run()
