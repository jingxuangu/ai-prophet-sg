"""
Custom Prophet Arena Trading Agent v5

v5 changes:
A. Switched the filter strategy to a blacklist to enlarge the candidate pool
B. Fixed the clamp dead zone so evidence_score = 0 still allows +/-3%
C. Added two-stage market screening (batch LLM pre-screen + deep analysis)
D. Trigger cooldown after consecutive skips
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

PA_API_URL            = "https://ai-prophet-core-api-998105805337.us-central1.run.app"
PA_API_KEY            = os.environ["PA_SERVER_API_KEY"]
OPENAI_KEY            = os.environ["OPENAI_API_KEY"]
BRAVE_KEY             = os.environ.get("BRAVE_API_KEY", "")

EXPERIMENT_SLUG       = "news_kelly_v5_exp3"
N_TICKS               = 1
STARTING_CASH         = 10_000.0
SCREEN_N_MARKETS      = 20
BATCH_SCREEN_CAP      = 50

MIN_EDGE              = 0.002
MIN_BET               = 10.0
MAX_PER_MARKET        = 150.0
SHRINKAGE             = 0.0
MAX_DELTA_FROM_MID    = 0.08
COOLDOWN_TICKS        = 3
BRIEF_LOG_FILE        = "brief_log.txt"
TRADED_IDS_FILE       = "traded_market_ids.json"
COOLDOWN_FILE         = "market_cooldown.json"

BLACKLIST = [
    # Sports - winners/awards
    "who will win", "who will attend", " mvp", "rookie of the year",
    "sixth man", "coach of the year", "championship", "stanley cup",
    "world cup", "eastern conference", "western conference", " pga ",
    "heisman", "eurovision winner",
    "pro basketball finals", "nba finals", "survivor season",
    "super bowl", "world series", "win the 2026", "win the 2027",
    # Crypto price levels
    "reach above $", "reach below $", "maximum wti",
    "bitcoin price", "ethereum price",
    "btc price", "btc above", "btc below",
    "bitcoin above", "bitcoin below", "bitcoin be above", "bitcoin be below",
    "eth price", "eth above", "eth below",
    # Long-horizon speculative elections
    "presidential nominee in 2028", "presidential nominee in 2029",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=OPENAI_KEY)


# ── Utility Functions ─────────────────────────────────────────────────────────

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
                time.sleep((2 ** attempt) + 0.5)
                logger.warning(f"OpenAI retry {attempt+1}: {e}")
            else:
                raise


def search_news(query: str, num_results: int = 5) -> str:
    if not BRAVE_KEY:
        return ""
    query = query[:200]
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/news/search",
            params={"q": query, "count": num_results, "freshness": "pw"},
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY},
            timeout=10,
        )
        if resp.status_code != 200:
            return ""
        seen, lines = set(), []
        for r in resp.json().get("results", []):
            title = (r.get("title") or "").strip()
            desc  = (r.get("description") or "").strip()
            age   = (r.get("age") or "").strip()
            if not title or title.lower() in seen:
                continue
            seen.add(title.lower())
            lines.append(f"- [{age}] {title}: {desc}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Search error: {e}")
        return ""


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
        ev = a.get("evidence_score")
        p_str = f"p={p_yes:.1%}" if p_yes is not None else "p=?"
        e_str = f"edge={edge:+.1%}" if edge is not None else ""
        ev_str = f"ev={ev}" if ev is not None else ""
        lines.append(f"  {i:2d}. [{p_str} {e_str} {ev_str}] {q}")
        lines.append(f"       → {a.get('decision', '')}")
    trades = record.get("trades", [])
    accepted, rejected = record.get("accepted", 0), record.get("rejected", 0)
    if trades:
        lines.append(f"{'─'*60}")
        lines.append(f"Trades executed ({accepted} accepted, {rejected} rejected):")
        for t in trades:
            lines.append(f"  BUY {t.get('side')}  shares={t.get('shares')}  market={t.get('market_id')}")
    else:
        lines.append(f"{'─'*60}")
        lines.append("No trades executed this tick.")
    with open(BRIEF_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Persistence: Traded IDs + Cooldown ───────────────────────────────────────

def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_traded_ids() -> set[str]:
    return set(_load_json(TRADED_IDS_FILE, []))


def save_traded_ids(ids: set[str]):
    _save_json(TRADED_IDS_FILE, sorted(ids))


def load_cooldown() -> dict:
    return _load_json(COOLDOWN_FILE, {})


def save_cooldown(state: dict):
    _save_json(COOLDOWN_FILE, state)


def is_on_cooldown(mid: str, state: dict, tick: int) -> bool:
    return tick < state.get(mid, {}).get("cooldown_until", 0)


def update_cooldown(state: dict, mid: str, was_skipped: bool, tick: int):
    info = state.get(mid, {"skip_streak": 0, "cooldown_until": 0})
    if was_skipped:
        info["skip_streak"] = info.get("skip_streak", 0) + 1
        if info["skip_streak"] >= 2:
            info["cooldown_until"] = tick + COOLDOWN_TICKS
            info["skip_streak"] = 0
            logger.info(f"    Cooldown: {mid} until tick {info['cooldown_until']}")
    else:
        info["skip_streak"] = 0
    state[mid] = info


# ── A. Blacklist Filtering ────────────────────────────────────────────────────

def passes_blacklist(question: str) -> bool:
    q = question.lower()
    return not any(p in q for p in BLACKLIST)


# ── C. Two-Stage Filtering ────────────────────────────────────────────────────

def build_candidate_pool(markets, cooldown_state, tick_count):
    """Stage 1: blacklist + price range + cooldown."""
    pool = []
    excluded_cd = 0
    for m in markets:
        yes_ask = float(m.quote.best_ask)
        if not (0.02 <= yes_ask <= 0.98):
            continue
        if not passes_blacklist(m.question):
            continue
        if is_on_cooldown(m.market_id, cooldown_state, tick_count):
            excluded_cd += 1
            continue
        pool.append(m)
    if excluded_cd:
        logger.info(f"Cooldown excluded {excluded_cd} markets")
    logger.info(f"Candidate pool: {len(pool)} markets")
    return pool


def batch_screen_markets(pool, n=SCREEN_N_MARKETS, tick_ts=""):
    """Stage 2: batch LLM pre-screening to pick the n markets most likely affected by recent news."""
    if len(pool) <= n:
        return pool

    # Truncate by spread to keep prompt size under control
    def spread(m):
        return float(m.quote.best_ask) - float(m.quote.best_bid)
    pool_sorted = sorted(pool, key=spread)[:BATCH_SCREEN_CAP]

    questions = []
    for i, m in enumerate(pool_sorted):
        yes_ask = float(m.quote.best_ask)
        yes_bid = float(m.quote.best_bid)
        mid = (yes_ask + yes_bid) / 2.0
        questions.append({
            "id": i,
            "q": m.question[:120],
            "mid": f"{mid:.0%}",
            "spread": f"{yes_ask - yes_bid:.1%}",
        })

    today = tick_ts[:10] if tick_ts else "2026-04-05"

    prompt = f"""Today is {today}. You are screening prediction markets for news-driven trading.

Select the {n} markets most likely to have exploitable mispricing due to recent news.

Prefer markets where:
- Discrete near-term events (visits, approvals, resignations, deals) have recent developments
- The current market price (mid) might not yet reflect very recent news
- The spread is small enough to trade profitably
- Topics are actively in the news right now

Avoid selecting:
- Long-term structural questions unlikely to change in the next few weeks
- Topics with no recent news activity
- Markets with very wide spreads

Markets (id, question, mid price, spread):
{json.dumps(questions, indent=2)}

Respond with ONLY:
{{"selected_ids": [list of up to {n} integer ids]}}"""

    try:
        result = call_openai(prompt, max_tokens=400)
        selected_ids = result.get("selected_ids", [])
        if not isinstance(selected_ids, list):
            raise ValueError("selected_ids not a list")
        selected_ids = [i for i in selected_ids if isinstance(i, int) and 0 <= i < len(pool_sorted)]
        selected = [pool_sorted[i] for i in selected_ids[:n]]
        logger.info(f"Batch screen: {len(pool)} → {len(pool_sorted)} (spread cap) → {len(selected)} (LLM)")
        return selected
    except Exception as e:
        logger.warning(f"Batch screen failed ({e}), fallback to top {n} by volume")
        return sorted(pool_sorted, key=lambda m: float(m.quote.volume_24h or 0), reverse=True)[:n]


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
        logger.warning(f"Query generation failed: {e}")
        return [question[:200]]


# ── Probability Forecasting (Core Model) ─────────────────────────────────────

def clamp_around_mid(raw_p: float, mid: float, max_delta: float) -> float:
    lower = max(0.01, mid - max_delta)
    upper = min(0.99, mid + max_delta)
    return max(lower, min(upper, raw_p))


def predict_market(market, memory_text: str = "") -> tuple[float, str, str, int]:
    """
    Returns: (adjusted_p_yes, rationale, news_text, evidence_score)
    evidence_score: 0 = none, 1 = weak, 2 = medium, 3 = strong
    """
    question   = market.question
    yes_ask    = float(market.quote.best_ask)
    yes_bid    = float(market.quote.best_bid)
    mid_price  = (yes_ask + yes_bid) / 2.0
    resolution = market.resolution_time.strftime("%Y-%m-%d")
    res_year   = market.resolution_time.year

    queries = generate_search_queries(question, res_year)
    all_news = []
    for q in queries:
        result = search_news(q)
        if result:
            all_news.append(f"[Query: {q}]\n{result}")

    news_block = "\nRecent news:\n" + "\n\n".join(all_news) if all_news else "\n(No recent news found)"
    memory_block = f"\n\nYour notes from previous ticks:\n{memory_text}" if memory_text else ""

    prompt = f"""You are a prediction market analyst. Decide if this market is mispriced.

Market question: {question}
Resolves: {resolution}

Current prices:
- To buy YES costs {yes_ask:.1%} (profitable if true probability > {yes_ask:.1%})
- To buy NO costs {1.0 - yes_bid:.1%} (profitable if true probability < {yes_bid:.1%})
- Mid: {mid_price:.1%}
{news_block}{memory_block}

Your task: Estimate the TRUE probability of YES resolution.

Evidence score:
- 0 = no direct recent evidence
- 1 = weak or indirect recent information
- 2 = clear recent evidence that directly affects resolution odds
- 3 = very strong direct evidence, close to decisive

Rules:
- If true probability > {yes_ask:.1%}, buying YES has positive edge.
- If true probability < {yes_bid:.1%}, buying NO has positive edge.
- If you see no edge, set p_yes close to mid ({mid_price:.1%}).
- Only move away from mid if you have actual evidence.

Respond with ONLY a JSON object:
{{
  "p_yes": <float 0-1>,
  "evidence_score": <integer 0-3>,
  "rationale": "<short explanation>"
}}"""

    raw = call_openai(prompt, max_tokens=300)

    raw_p = max(0.01, min(0.99, float(raw["p_yes"])))
    rationale = raw.get("rationale", "")
    ev_score = max(0, min(3, int(raw.get("evidence_score", 0))))

    # Global clamp
    raw_p = clamp_around_mid(raw_p, mid_price, MAX_DELTA_FROM_MID)

    # B. Clamp by evidence strength (score = 0 -> +/-3%, fixing the dead zone)
    ev_cap = {0: 0.03, 1: 0.04, 2: 0.06, 3: 0.10}[ev_score]
    raw_p = clamp_around_mid(raw_p, mid_price, ev_cap)

    adjusted = (1.0 - SHRINKAGE) * raw_p + SHRINKAGE * mid_price

    logger.info(
        f"    p_yes={adjusted:.1%} (mid={mid_price:.1%}, ev={ev_score}, cap=±{ev_cap:.0%})"
    )
    return adjusted, rationale, "\n\n".join(all_news), ev_score


# ── Position Sizing ───────────────────────────────────────────────────────────

def compute_edge_and_side(p_yes, yes_ask, no_ask):
    yes_edge = p_yes - yes_ask
    no_edge  = (1.0 - p_yes) - no_ask
    if yes_edge >= no_edge:
        return "YES", max(yes_edge, 0.0)
    return "NO", max(no_edge, 0.0)


def fixed_size_from_edge(edge):
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
    # --- Setup (once) ---
    api = ServerAPIClient(base_url=PA_API_URL, api_key=PA_API_KEY)

    exp = api.create_or_get_experiment(
        slug=EXPERIMENT_SLUG,
        config_hash="news-kelly-v5-blacklist-twostage",
        config_json={
            "strategy": "news_search_v5",
            "model": "gpt-4o",
            "improvements": [
                "blacklist_filter",
                "two_stage_screening",
                "clamp_dead_zone_fix",
                "cooldown_on_skip",
                "cross_tick_memory",
            ],
            "shrinkage": SHRINKAGE,
            "max_delta_from_mid": MAX_DELTA_FROM_MID,
            "cooldown_ticks": COOLDOWN_TICKS,
        },
        n_ticks=N_TICKS,
    )
    experiment_id = exp.experiment_id
    logger.info(f"Experiment: {experiment_id} | slug={EXPERIMENT_SLUG} | new={exp.created}")

    part = api.upsert_participant(
        experiment_id, model="custom:news-gpt4o-v5", rep=0, starting_cash=STARTING_CASH,
    )
    participant_idx = part.participant_idx
    logger.info(f"Participant idx: {participant_idx}")

    # --- Tick loop ---
    lease_owner = str(uuid.uuid4())
    tick_count = 0
    prev_tick_record: Optional[dict] = None
    cooldown_state = load_cooldown()

    while True:
        claim = api.claim_tick(experiment_id, lease_owner)

        if claim.no_tick_available:
            if claim.reason == "experiment_completed":
                logger.info(f"Experiment complete after {tick_count} ticks.")
                break
            time.sleep(claim.retry_after_sec or 15)
            continue

        tick_id = claim.tick_id
        snapshot_id = claim.snapshot_id
        tick_count += 1
        logger.info(f"\n{'='*60}\nTick {tick_count} | {tick_id}\n{'='*60}")

        try:
            # ── Portfolio ─────────────────────────────────────────────
            portfolio = api.get_portfolio(experiment_id, participant_idx)
            if portfolio:
                cash   = float(portfolio.cash)
                equity = float(portfolio.equity)
                pnl    = float(portfolio.total_pnl)
                for pos in (portfolio.positions or []):
                    logger.info(f"  Position: {pos.market_id} | shares={pos.shares} | value={pos.market_value}")
                logger.info(f"Portfolio: cash=${cash:,.2f} | equity=${equity:,.2f} | pnl=${pnl:+,.2f}")
            else:
                cash, equity, pnl = STARTING_CASH, STARTING_CASH, 0.0
                logger.warning("get_portfolio returned None — using defaults")

            # ── Get markets & screen ──────────────────────────────────
            candidates = api.get_candidates(claim.tick_ts, snapshot_id)
            pool = build_candidate_pool(candidates.markets, cooldown_state, tick_count)
            tick_ts_str = str(claim.tick_ts) if claim.tick_ts else ""
            top_markets = batch_screen_markets(pool, n=SCREEN_N_MARKETS, tick_ts=tick_ts_str)

            memory_text = build_memory_text(prev_tick_record)

            tick_record = {
                "tick_id": tick_id, "tick_count": tick_count,
                "cash": cash, "equity": equity, "pnl": pnl,
                "analyzed": [], "trades": [], "accepted": 0, "rejected": 0,
            }

            # ── Analyze each market & build intents ───────────────────
            intents = []

            for i, market in enumerate(top_markets):
                market_id = market.market_id
                yes_ask = float(market.quote.best_ask)
                yes_bid = float(market.quote.best_bid)
                no_ask  = 1.0 - yes_bid

                logger.info(
                    f"  [{i+1}/{len(top_markets)}] {market.question[:70]}"
                    f"\n    YES={yes_ask:.1%}  NO={no_ask:.1%}  vol24h=${market.quote.volume_24h:,.0f}"
                )

                market_log = {
                    "market_id": market_id,
                    "question": market.question,
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                }

                try:
                    p_yes, rationale, _, ev_score = predict_market(market, memory_text)
                    side, edge = compute_edge_and_side(p_yes, yes_ask, no_ask)
                    amount = fixed_size_from_edge(edge)

                    market_log.update({
                        "p_yes": round(p_yes, 4), "rationale": rationale,
                        "edge": round(edge, 4), "side": side,
                        "amount": round(amount, 2), "evidence_score": ev_score,
                    })

                    logger.info(
                        f"    p_yes={p_yes:.1%} | edge={edge:+.1%} | ev={ev_score} | "
                        f"size=${amount:.0f} | {rationale}"
                    )

                    # Decide whether to trade
                    if edge < MIN_EDGE:
                        market_log["decision"] = f"SKIP: edge {edge:.1%} < min {MIN_EDGE:.1%}"
                        update_cooldown(cooldown_state, market_id, True, tick_count)
                        tick_record["analyzed"].append(market_log)
                        continue

                    if amount < MIN_BET:
                        market_log["decision"] = f"SKIP: size ${amount:.0f} < min ${MIN_BET}"
                        update_cooldown(cooldown_state, market_id, True, tick_count)
                        tick_record["analyzed"].append(market_log)
                        continue

                    market_log["decision"] = f"TRADE: BUY {side} ${amount:.0f}"
                    update_cooldown(cooldown_state, market_id, False, tick_count)

                    intents.append(TradeIntentRequest(
                        market_id=market_id,
                        action="BUY",
                        side=side,
                        shares=str(int(amount)),
                        idempotency_key=f"{experiment_id}:{participant_idx}:{tick_id}:{i}",
                    ))

                except Exception as e:
                    market_log["decision"] = f"ERROR: {e}"
                    logger.warning(f"    Prediction failed for {market_id}: {e}")

                tick_record["analyzed"].append(market_log)

            save_cooldown(cooldown_state)

            # ── Submit trades ─────────────────────────────────────────
            if intents:
                result = api.submit_trade_intents(
                    experiment_id, participant_idx, tick_id,
                    candidates.candidate_set_id, intents,
                )
                tick_record["accepted"] = result.accepted
                tick_record["rejected"] = result.rejected
                tick_record["trades"] = [
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
                logger.info("No trades this tick")

            write_brief_log(tick_record)
            prev_tick_record = tick_record

            # ── Finalize tick ─────────────────────────────────────────
            api.finalize_participant(experiment_id, participant_idx, tick_id, status="COMPLETED")
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

    save_cooldown(cooldown_state)
    api.close()
    logger.info("Agent stopped.")


if __name__ == "__main__":
    logger.info("Starting News-Driven Kelly Agent v5")
    logger.info(f"Brave Search: {'ENABLED' if BRAVE_KEY else 'DISABLED'}")
    logger.info(f"Shrinkage: {SHRINKAGE} | Cooldown: {COOLDOWN_TICKS} ticks")
    run()
