"""
Custom Prophet Arena Trading Agent v3

v3 goals:
1. Be more conservative: use the market price as the base and let the LLM make only modest adjustments
2. Filter out markets that are not well suited for a news-driven strategy
3. Avoid contamination from stale or mismatched news
4. Use fixed tiered sizing to validate signal quality first
5. Preserve the overall framework with minimal structural changes
"""

import json
import logging
import os
import time
import uuid
from typing import Optional

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

EXPERIMENT_SLUG       = "news_kelly_t5"
N_TICKS               = 2
STARTING_CASH         = 10_000.0
TOP_N_MARKETS         = 10

MIN_EDGE              = 0.01
MIN_BET               = 20.0
MAX_PER_MARKET        = 150.0
SHRINKAGE             = 0.0
MAX_DELTA_FROM_MID    = 0.08   # LLM can deviate from the market by at most 8 percentage points
BRIEF_LOG_FILE        = "brief_log.txt"
TRADED_IDS_FILE       = "traded_market_ids.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_KEY)


# ── Logging ───────────────────────────────────────────────────────────────────

def write_brief_log(record: dict) -> None:
    lines = [
        f"\n{'='*60}",
        f"Tick #{record.get('tick_count')}  id={record.get('tick_id')}",
        f"Cash=${record.get('cash', 0):,.0f}  Equity=${record.get('equity', 0):,.0f}  PnL=${record.get('pnl', 0):+,.0f}",
        f"{'─'*60}",
        "Selected markets:",
    ]

    for i, a in enumerate(record.get("analyzed", []), 1):
        q = a.get("question", "")[:70]
        p_yes = a.get("p_yes")
        edge = a.get("edge")
        decision = a.get("decision", "")
        p_str = f"p={p_yes:.1%}" if p_yes is not None else "p=?"
        e_str = f"edge={edge:+.1%}" if edge is not None else ""
        lines.append(f"  {i:2d}. [{p_str} {e_str}] {q}")
        lines.append(f"       → {decision}")

    trades = record.get("trades", [])
    accepted = record.get("accepted", 0)
    rejected = record.get("rejected", 0)
    if trades:
        lines.append(f"{'─'*60}")
        lines.append(f"Trades executed ({accepted} accepted, {rejected} rejected):")
        for t in trades:
            lines.append(
                f"  BUY {t.get('side')}  shares={t.get('shares')}  market={t.get('market_id')}"
            )
    else:
        lines.append(f"{'─'*60}")
        lines.append("No trades executed this tick.")

    with open(BRIEF_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Brief log written → {BRIEF_LOG_FILE}")


# ── OpenAI Calls ──────────────────────────────────────────────────────────────

def call_openai(prompt: str, max_retries: int = 3, **kwargs) -> dict:
    kwargs.setdefault("max_tokens", 200)

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                **kwargs,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + 0.5
                logger.warning(
                    f"OpenAI call failed (attempt {attempt+1}): {e}, retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                raise


# ── News Search ───────────────────────────────────────────────────────────────

def search_news(query: str, num_results: int = 5) -> str:
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
        seen_titles = set()

        for r in results:
            title = (r.get("title") or "").strip()
            desc  = (r.get("description") or "").strip()
            age   = (r.get("age") or "").strip()

            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())

            lines.append(f"- [{age}] {title}: {desc}")

        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Search error for '{query}': {e}")
        return ""


# ── Market Filtering ──────────────────────────────────────────────────────────


def classify_market_priority(question: str) -> tuple[bool, int]:
    """
    Returns:
    - Whether the market should enter the news-driven pool
    - A priority score, where higher means more preferred

    Goals:
    1. Prioritize explicit event-driven markets
    2. Downweight long-horizon price-threshold markets
    3. Treat low volume as a secondary ranking factor rather than the core signal
    """
    q = question.lower().strip()

    # Clearly unsuitable for the current news-driven framework
    hard_exclude = [
        "who will win",
        "who will attend",
        "mvp",
        "rookie of the year",
        "sixth man",
        "coach of the year",
        "championship",
        "stanley cup",
        "world cup",
        "eastern conference",
        "western conference",
        "pga",
        "heisman",
        "eurovision winner",
        "price on",
    ]
    if any(p in q for p in hard_exclude):
        return False, -999

    score = 0

    # Tier A: explicit event markets best suited for news-driven trading
    strong_event_markers = [
        "leave office",
        "leave prime minister",
        "change its ceo",
        "be approved",
        "be confirmed",
        "becomes law",
        "become law",
        "visit ",
        "agree to",
        "recognize ",
        "issue a level 4",
        "shutdown",
        "resign",
        "removed",
    ]
    for marker in strong_event_markers:
        if marker in q:
            score += 6

    # Tier B: moderately suitable markets
    medium_event_markers = [
        "cut rates",
        "be the head of state",
        "head of state",
        "government of",
        "nominee",
        "control be",
        "house control",
        "senate control",
        "reactor",
        "recession",
    ]
    for marker in medium_event_markers:
        if marker in q:
            score += 3

    # Tier C: long-horizon price/threshold markets, downweighted but not fully excluded
    weak_markers = [
        "reach above",
        "reach below",
        "maximum wti",
        "bitcoin",
        "ethereum",
        "$",
        "price",
    ]
    for marker in weak_markers:
        if marker in q:
            score -= 4

    # Base score
    if q.startswith("will "):
        score += 1

    if score <= 0:
        return False, score

    return True, score


# ── Traded Market Tracking (Lightweight Persistence) ──────────────────────────

def load_traded_market_ids() -> set[str]:
    if not os.path.exists(TRADED_IDS_FILE):
        return set()
    try:
        with open(TRADED_IDS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    except Exception as e:
        logger.warning(f"Could not load traded IDs: {e}")
        return set()


def save_traded_market_ids(ids: set[str]) -> None:
    with open(TRADED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f)


def select_top_markets(markets: list, n: int = TOP_N_MARKETS) -> list:
    already_traded = load_traded_market_ids()
    if already_traded:
        logger.info(f"Excluding {len(already_traded)} already-traded markets")

    ranked = []
    for m in markets:
        yes_ask = float(m.quote.best_ask)
        vol = float(m.quote.volume_24h or 0)

        if m.market_id in already_traded:
            continue
        if not (0.02 <= yes_ask <= 0.98):
            continue

        ok, priority = classify_market_priority(m.question)
        if not ok:
            continue

        ranked.append((priority, vol, m))

    # Sort first by event priority, then by lower trading volume
    ranked.sort(key=lambda x: (-x[0], x[1]))

    selected = [m for _, _, m in ranked[:n]]

    logger.info(
        f"Hunting Mode t3: {len(ranked)} ranked news-driven markets found. Top {min(n, len(selected))} selected."
    )
    return selected


# ── Search Query Generation ───────────────────────────────────────────────────

def generate_search_queries(question: str, resolution_year: int) -> list[str]:
    prompt = f"""Convert this prediction market question into 2 short RECENT-news web search queries.

Important rules:
- Prefer recent-news queries anchored to {resolution_year} or 2026/2027 if relevant.
- Do NOT use old years like 2023 unless the market itself is about that year.
- Focus on direct, resolution-relevant evidence.
- Keep each query short.

Question: {question}

Respond with ONLY a JSON object:
{{"queries": ["query1", "query2"]}}"""

    try:
        result = call_openai(prompt, max_tokens=120)
        queries = result.get("queries", [])
        if not queries or not isinstance(queries, list):
            return [question[:200]]
        return [q[:200] for q in queries[:2]]
    except Exception as e:
        logger.warning(f"Query generation failed: {e}, falling back to raw question")
        return [question[:200]]


# ── Probability Forecasting ───────────────────────────────────────────────────

def clamp_probability_around_mid(raw_p: float, mid_price: float, max_delta: float) -> float:
    lower = max(0.01, mid_price - max_delta)
    upper = min(0.99, mid_price + max_delta)
    return max(lower, min(upper, raw_p))


def predict_market(market, memory_text: str = "") -> tuple[float, str, str, int]:
    """
    Returns:
    - adjusted_p_yes
    - rationale
    - news_text
    - evidence_score (0-3)

    evidence_score:
    0 = no direct evidence / pure background
    1 = weak but relevant recent information
    2 = clear and fairly direct recent evidence
    3 = strong direct evidence, close to decisive
    """
    question      = market.question
    yes_ask       = float(market.quote.best_ask)
    yes_bid       = float(market.quote.best_bid)
    mid_price     = (yes_ask + yes_bid) / 2.0
    resolution    = market.resolution_time.strftime("%Y-%m-%d")
    resolution_yr = market.resolution_time.year

    queries = generate_search_queries(question, resolution_yr)

    all_news = []
    for q in queries:
        result = search_news(q)
        if result:
            all_news.append(f"[Query: {q}]\n{result}")

    news_block = "\nRecent news:\n" + "\n\n".join(all_news) if all_news else "\n(No recent news found)"
    memory_block = f"\n\nYour notes from previous ticks:\n{memory_text}" if memory_text else ""

    prompt = f"""You are a cautious analyst checking whether a low-attention prediction market may be stale.

Market question: {question}
Resolves: {resolution}
Current market implied probability: {mid_price:.1%}
{news_block}{memory_block}

Your tasks:
1. Estimate fair probability.
2. Judge evidence strength.

Evidence score rules:
- 0 = no direct recent evidence; mostly background or speculation
- 1 = weak but somewhat relevant recent information
- 2 = clear recent evidence that directly affects resolution odds
- 3 = very strong direct evidence, close to decisive

Important rules:
- Default to market being roughly right unless there is direct recent evidence otherwise.
- Do NOT make bold adjustments from vague or indirect news.
- Weak evidence should imply only tiny changes.
- Strong evidence is required for larger deviations.
- If there is no clear direct evidence, stay very near the market price.

Respond with ONLY a JSON object:
{{
  "p_yes": <float 0-1>,
  "evidence_score": <integer 0-3>,
  "rationale": "<short explanation>"
}}"""

    raw = call_openai(prompt, max_tokens=300)

    raw_p_yes = max(0.01, min(0.99, float(raw["p_yes"])))
    rationale = raw.get("rationale", "")
    evidence_score = int(raw.get("evidence_score", 0))
    evidence_score = max(0, min(3, evidence_score))

    # First apply a global clamp so the model cannot drift too far from the market
    raw_p_yes = clamp_probability_around_mid(
        raw_p_yes,
        mid_price,
        MAX_DELTA_FROM_MID,
    )

    # Then cap the maximum deviation based on evidence strength
    evidence_delta_cap = {
        0: 0.01,   # Almost pinned to the market
        1: 0.03,   # Weak evidence, only a small adjustment
        2: 0.06,   # Medium evidence, a moderate adjustment
        3: 0.10,   # Strong evidence, allow a larger adjustment
    }[evidence_score]

    raw_p_yes = clamp_probability_around_mid(
        raw_p_yes,
        mid_price,
        evidence_delta_cap,
    )

    # Finally apply shrinkage
    adjusted_p_yes = (1.0 - SHRINKAGE) * raw_p_yes + SHRINKAGE * mid_price

    logger.info(
        f"    p_yes: raw={raw_p_yes:.1%} → shrunk={adjusted_p_yes:.1%} "
        f"(market mid={mid_price:.1%}, evidence={evidence_score}, shrinkage={SHRINKAGE})"
    )

    return adjusted_p_yes, rationale, "\n\n".join(all_news), evidence_score


# ── Position Sizing ───────────────────────────────────────────────────────────

def compute_edge_and_side(p_yes: float, yes_ask: float, no_ask: float) -> tuple[str, float]:
    yes_edge = p_yes - yes_ask
    no_edge  = (1.0 - p_yes) - no_ask
    if yes_edge >= no_edge:
        return "YES", max(yes_edge, 0.0)
    return "NO", max(no_edge, 0.0)


def fixed_size_from_edge(edge: float) -> float:
    if edge < MIN_EDGE:
        return 0.0
    elif edge < 0.05:
        return 50.0
    elif edge < 0.08:
        return 100.0
    else:
        return min(150.0, MAX_PER_MARKET)


# ── Cross-Tick Memory ─────────────────────────────────────────────────────────

def build_memory_text(prev_record: Optional[dict]) -> str:
    if not prev_record:
        return ""

    lines = [
        f"Previous tick: {prev_record.get('tick_id', '?')}",
        f"Portfolio: cash=${prev_record.get('cash', 0):,.0f}, pnl=${prev_record.get('pnl', 0):+,.0f}",
    ]

    analyzed = prev_record.get("analyzed", [])
    traded = [a for a in analyzed if str(a.get("decision", "")).startswith("TRADE")]
    skipped = [a for a in analyzed if str(a.get("decision", "")).startswith("SKIP")]

    if traded:
        lines.append("Trades made:")
        for t in traded:
            lines.append(
                f"  - {t['market_id']}: {t['side']} ${t.get('amount', 0):.0f} "
                f"(p_yes={t.get('p_yes', 0):.1%}, edge={t.get('edge', 0):+.1%})"
            )

    if skipped:
        lines.append(f"Skipped {len(skipped)} markets")

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
        config_hash="news-kelly-t2-conservative",
        config_json={
            "strategy": "news_search_t2_conservative",
            "model": "gpt-4o",
            "improvements": [
                "market_filtering_news_driven_only",
                "recent_query_guardrails",
                "probability_clamp",
                "probability_shrinkage",
                "fixed_position_sizing",
                "retry_logic",
                "cross_tick_memory",
            ],
            "shrinkage": SHRINKAGE,
            "max_delta_from_mid": MAX_DELTA_FROM_MID,
        },
        n_ticks=N_TICKS,
    )
    experiment_id = exp.experiment_id
    logger.info(f"Experiment: {experiment_id} | slug={EXPERIMENT_SLUG} | new={exp.created}")

    part = api.upsert_participant(
        experiment_id,
        model="custom:news-gpt4o-t2",
        rep=0,
        starting_cash=STARTING_CASH,
    )
    participant_idx = part.participant_idx
    logger.info(f"Participant idx: {participant_idx}")

    lease_owner = str(uuid.uuid4())
    tick_count = 0
    prev_tick_record: Optional[dict] = None

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

        tick_id = claim.tick_id
        snapshot_id = claim.snapshot_id
        tick_count += 1

        logger.info(f"\n{'='*60}\nTick {tick_count} | {tick_id}\n{'='*60}")

        try:
            portfolio = api.get_portfolio(experiment_id, participant_idx)
            if portfolio:
                cash = float(portfolio.cash)
                equity = float(portfolio.equity)
                pnl = float(portfolio.total_pnl)
                for pos in (portfolio.positions or []):
                    logger.info(f"  Position: {pos.market_id} | shares={pos.shares} | value={pos.market_value}")
                logger.info(
                    f"Portfolio: cash=${cash:,.2f} | equity=${equity:,.2f} | pnl=${pnl:+,.2f}"
                )
            else:
                cash = STARTING_CASH
                equity = STARTING_CASH
                pnl = 0.0
                logger.warning("get_portfolio returned None (404) — using defaults")

            candidates = api.get_candidates(claim.tick_ts, snapshot_id)
            top_markets = select_top_markets(candidates.markets)

            memory_text = build_memory_text(prev_tick_record)
            if memory_text:
                logger.info(f"Memory from previous tick: {len(memory_text)} chars")

            tick_record = {
                "tick_id": tick_id,
                "tick_count": tick_count,
                "cash": cash,
                "equity": equity,
                "pnl": pnl,
                "analyzed": [],
                "trades": [],
                "accepted": 0,
                "rejected": 0,
            }

            intents = []

            for i, market in enumerate(top_markets):
                market_id = market.market_id
                yes_ask = float(market.quote.best_ask)
                yes_bid = float(market.quote.best_bid)
                no_ask = 1.0 - yes_bid

                logger.info(
                    f"  [{i+1}/{len(top_markets)}] {market.question[:70]}"
                    f"\n    Prices: YES={yes_ask:.1%}  NO={no_ask:.1%}  vol24h=${market.quote.volume_24h:,.0f}"
                )

                market_log = {
                    "market_id": market_id,
                    "question": market.question,
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                }

                try:
                    p_yes, rationale, news_found, evidence_score = predict_market(
                        market, memory_text=memory_text)

                    side, edge = compute_edge_and_side(p_yes, yes_ask, no_ask)
                    amount = fixed_size_from_edge(edge)

                    market_log.update({
                        "p_yes": round(p_yes, 4),
                        "rationale": rationale,
                        "edge": round(edge, 4),
                        "side": side,
                        "amount": round(amount, 2),
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

                    intents.append(
                        TradeIntentRequest(
                            market_id=market_id,
                            action="BUY",
                            side=side,
                            shares=str(int(amount)),
                            idempotency_key=f"{experiment_id}:{participant_idx}:{tick_id}:{i}",
                        )
                    )

                except Exception as e:
                    market_log["decision"] = f"ERROR: {e}"
                    logger.warning(f"    Prediction failed for {market_id}: {e}")

                tick_record["analyzed"].append(market_log)

            if intents:
                result = api.submit_trade_intents(
                    experiment_id,
                    participant_idx,
                    tick_id,
                    candidates.candidate_set_id,
                    intents,
                )
                tick_record["accepted"] = result.accepted
                tick_record["rejected"] = result.rejected
                tick_record["trades"] = [
                    {
                        "market_id": r.market_id,
                        "action": r.action,
                        "side": r.side,
                        "shares": r.shares,
                    }
                    for r in result.fills
                ]

                # Persist traded market IDs
                traded_ids = load_traded_market_ids()
                for fill in result.fills:
                    traded_ids.add(fill.market_id)
                save_traded_market_ids(traded_ids)

                logger.info(
                    f"Submitted {len(intents)} intents → "
                    f"{result.accepted} accepted, {result.rejected} rejected"
                )
                for r in result.rejections:
                    logger.warning(f"  Rejected: {r.reason}")
            else:
                logger.info("No trades this tick (no edge found)")

            write_brief_log(tick_record)
            prev_tick_record = tick_record

            api.finalize_participant(
                experiment_id,
                participant_idx,
                tick_id,
                status="COMPLETED",
            )
            api.complete_tick(experiment_id, tick_id)
            logger.info(f"Tick {tick_id} ✓")

        except Exception as e:
            logger.error(f"Tick failed: {e}", exc_info=True)
            try:
                api.finalize_participant(
                    experiment_id,
                    participant_idx,
                    tick_id,
                    status="FAILED",
                    error_detail=str(e)[:500],
                )
                api.complete_tick(experiment_id, tick_id)
            except Exception:
                pass

    api.close()
    logger.info("Agent stopped.")


if __name__ == "__main__":
    logger.info("Starting News-Driven Kelly Agent t2")
    logger.info(f"Brave Search: {'ENABLED' if BRAVE_KEY else 'DISABLED (no key)'}")
    logger.info(f"Shrinkage: {SHRINKAGE}")
    run()
