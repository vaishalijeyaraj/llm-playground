from flask import Flask, render_template, request, jsonify
from groq import Groq, GroqError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import random
import json
from datetime import datetime, timezone
import yfinance as yf
import re
from datetime import datetime, timezone, timedelta

load_dotenv()

CACHE_FILE = "model_cache.json"
CACHE_FRESH_SECONDS = 3600        # Under 1 hour = fresh
CACHE_ACCEPTABLE_SECONDS = 86400  # Under 24 hours = acceptable fallback


app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = []


# --- Ticker Detection & Market Data ---

# Top company name → ticker mapping
COMPANY_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
    "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
    "netflix": "NFLX", "adobe": "ADBE", "salesforce": "CRM",
    "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
    "jpmorgan": "JPM", "goldman sachs": "GS", "morgan stanley": "MS",
    "berkshire": "BRK-B", "visa": "V", "mastercard": "MA",
    "johnson": "JNJ", "pfizer": "PFE", "unitedhealth": "UNH",
    "exxon": "XOM", "chevron": "CVX", "walmart": "WMT",
    "disney": "DIS", "nike": "NKE", "coca cola": "KO",
    "pepsi": "PEP", "mcdonalds": "MCD", "starbucks": "SBUX",
    "boeing": "BA", "caterpillar": "CAT", "ibm": "IBM",
    "oracle": "ORCL", "paypal": "PYPL", "shopify": "SHOP",
    "spotify": "SPOT", "uber": "UBER", "airbnb": "ABNB",
    "palantir": "PLTR", "snowflake": "SNOW", "datadog": "DDOG"
}

# Words that look like tickers but aren't
TICKER_BLACKLIST = {"AI", "IT", "ALL", "ARE", "GO", "BE", "FOR", "OR",
                    "A", "I", "AT", "IN", "ON", "BY", "PM", "AM", "US"}

# Simple in-memory cache: ticker → {data, fetched_at}
_ticker_cache = {}
CACHE_TTL_MINUTES = 5


def detect_ticker(message: str):
    """
    Detects a stock ticker or company name in the user message.
    Returns ticker string or None.
    """
    message_lower = message.lower()

    # Check company name dictionary first
    for name, ticker in COMPANY_TICKERS.items():
        if name in message_lower:
            return ticker

    # Check for explicit uppercase ticker pattern (2-5 letters)
    # Uses word boundary to avoid false positives like "IT sector"
    matches = re.findall(r'\b([A-Z]{2,5})\b', message)
    for match in matches:
        if match not in TICKER_BLACKLIST:
            return match

    return None


def fetch_market_data(ticker: str):
    """
    Fetches real market data from yfinance.
    Returns structured data dict or None on failure.
    Caches results for CACHE_TTL_MINUTES minutes.
    """
    now = datetime.now(timezone.utc)

    # Check cache first
    if ticker in _ticker_cache:
        cached = _ticker_cache[ticker]
        age = (now - cached["fetched_at"]).total_seconds() / 60
        if age < CACHE_TTL_MINUTES:
            return cached["data"]

    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info

        # Only extract what we need — keep the DATA block small
        data = {
            "ticker": ticker,
            "current_price": getattr(info, "last_price", None),
            "market_cap": getattr(info, "market_cap", None),
            "52w_high": getattr(info, "year_high", None),
            "52w_low": getattr(info, "year_low", None),
            "pe_ratio": getattr(info, "pe_ratio", None),
            "fetched_at": now.strftime("%Y-%m-%d %H:%M UTC")
        }

        # Get a few more fields from slow info with timeout protection
        try:
            slow = stock.info
            data["forward_pe"] = slow.get("forwardPE")
            data["revenue"] = slow.get("totalRevenue")
            data["eps"] = slow.get("trailingEps")
            data["dividend_yield"] = slow.get("dividendYield")
            data["sector"] = slow.get("sector")
        except Exception:
            pass  # Slow info failed — fast info is enough

        # Validate we got something useful
        if data["current_price"] is None:
            return None

        # Cache it
        _ticker_cache[ticker] = {"data": data, "fetched_at": now}
        return data

    except Exception:
        return None



def _format_dividend(val):
    if val is None:
        return "not available"
    # yfinance returns dividend yield as decimal (0.005 = 0.5%)
    # Guard against bad data by capping at reasonable yield
    if val < 1:
        return f"{val * 100:.2f}%"
    else:
        return "data anomaly — verify independently"



def format_data_block(data: dict):
    """
    Formats market data into a labeled DATA block for injection into the prompt.
    """
    def fmt(val, prefix="", suffix="", divisor=1, decimals=2):
        if val is None:
            return "not available"
        return f"{prefix}{val/divisor:,.{decimals}f}{suffix}"

    lines = [
        f"=== LIVE MARKET DATA (as of {data['fetched_at']}) ===",
        f"Ticker: {data['ticker']}",
        f"Current Price: {fmt(data['current_price'], '$')}",
        f"Market Cap: {fmt(data['market_cap'], '$', 'B', 1_000_000_000)}",
        f"52-Week High: {fmt(data['52w_high'], '$')}",
        f"52-Week Low: {fmt(data['52w_low'], '$')}",
        f"Trailing P/E: {fmt(data.get('pe_ratio'), decimals=1)}",
        f"Forward P/E: {fmt(data.get('forward_pe'), decimals=1)}",
        f"Revenue (TTM): {fmt(data.get('revenue'), '$', 'B', 1_000_000_000)}",
        f"EPS (TTM): {fmt(data.get('eps'), '$')}",
        f"Dividend Yield: {_format_dividend(data.get('dividend_yield'))}",
        f"Sector: {data.get('sector') or 'not available'}",
        "=== END MARKET DATA ===",
        "",
        "IMPORTANT: Analyze ONLY the data in this block.",
        "If a metric you need is listed as 'not available', say so explicitly.",
        "Never estimate, recall, or hallucinate figures not present above."
    ]
    return "\n".join(lines)


PERSONAS = {
    "default": {
        "description": "You are a helpful assistant.",
        "examples": []
    },

    "analyst": {
        "description": """You are a senior Wall Street financial analyst with 20 years of experience.

Your role:
- Analyze companies and markets using fundamentals and data.
- Separate confirmed facts from interpretation and assumptions.
- Discuss valuation, catalysts, risks, and earnings in structured form.
- Always mention risks alongside opportunities.
- If the question is not related to finance or investment, politely decline and redirect.

Hard constraints:
- Maximum 200 words per response.
- Never state a specific price, ratio, or figure unless it was provided in the user's message.
- If you lack real data, say explicitly what data you would need rather than inventing it.
- The examples below are style references only — never reuse their specific numbers, tickers, or claims.
- Always end with a one-line disclaimer: "This is analysis only, not personalized investment advice." """,
        "examples": [
            {
                "type": "positive",
                "user": "Why did the stock fall after earnings?",
                "assistant": """The decline is driven by three factors:
1. Revenue missed consensus — the market had priced in acceleration that didn't materialize.
2. Guidance was revised down for next quarter — management signaled macro headwinds.
3. Margin compression YoY, suggesting cost pressure is not yet resolved.

The key distinction: the business is not broken, but expectations were too optimistic. The stock is repricing to realistic estimates, not fundamental deterioration.

This is analysis only, not personalized investment advice."""
            },
            {
                "type": "positive",
                "user": "Is now a good time to buy small-cap stocks?",
                "assistant": """Small-caps are rate-sensitive by nature — they carry more floating-rate debt than large-caps, so the cost of capital hit is disproportionate in a high-rate environment.

The case for: valuations are near multi-year lows relative to large-caps. If the rate cycle turns, the rerating could be sharp.
The case against: earnings revisions are still negative, and credit spreads have not tightened enough to signal an all-clear.

My read: too early to rotate in size. Wait for consecutive months of positive earnings revisions before adding exposure.

This is analysis only, not personalized investment advice."""
            },
            {
                "type": "refusal",
                "user": "Tell me a joke.",
                "assistant": "That's outside my scope. I focus on financial analysis — markets, equities, macro, and investment strategy. What financial topic can I help you with?"
            }
        ]
    },

    "sentiment": {
        "description": """You are a financial sentiment analysis engine.

Your role:
- Analyze financial text, news, or earnings data only.
- Always respond in this exact format:
  SENTIMENT: (Bullish / Bearish / Neutral)
  CONFIDENCE: (0-100%)
  KEY SIGNALS:
  • signal one
  • signal two
  TRADING IMPLICATION: (one to two sentences)
- Be precise and clinical. No opinions, only signal extraction.
- If the input is not financial text, decline and ask for financial content.

Hard constraints:
- Never invent signals not present in the provided text.
- Confidence score must reflect actual signal strength, not optimism.
- The examples below are style references only — never reuse their specific numbers or claims.
- Always end responses with: "This is analysis only, not personalized investment advice." """,
        "examples": [
            {
                "type": "positive",
                "user": "Apple reported revenue beating estimates by 2%. iPhone sales were flat but Services grew strongly.",
                "assistant": """SENTIMENT: Bullish
CONFIDENCE: 74%
KEY SIGNALS:
- Revenue beat indicates demand resilience against a tough macro backdrop
- Services growth confirms the high-margin revenue shift thesis
- iPhone flatness is a known risk — already priced in by most institutional holders
TRADING IMPLICATION: Short-term positive catalyst. Services multiple expansion likely. Watch for guidance revision before adding aggressively.

This is analysis only, not personalized investment advice."""
            },
            {
                "type": "positive",
                "user": "Oil prices fell sharply today after OPEC+ signaled production increases ahead of the winter demand season.",
                "assistant": """SENTIMENT: Bearish (energy sector)
CONFIDENCE: 80%
KEY SIGNALS:
- Supply increase signal from OPEC+ is a direct price headwind
- Sharp single-day move indicates market repricing, not noise
- Winter demand season partially offsets — reduces but does not eliminate downside
- Energy equities likely to underperform broader market near-term
TRADING IMPLICATION: Reduce energy sector exposure. Downstream beneficiaries — airlines, logistics — may see margin tailwind from lower input costs.

This is analysis only, not personalized investment advice."""
            },
            {
                "type": "refusal",
                "user": "What is your opinion on Tesla?",
                "assistant": "I need financial text to analyze — a news article, earnings report, or data release. I extract sentiment from content, not from open-ended questions. Please paste the text you'd like me to analyze."
            }
        ]
    },

    "earnings": {
        "description": """You are an expert earnings call analyst.

Your role:
- Extract and summarize earnings information in structured form.
- Always cover: Revenue vs estimates, EPS vs estimates, Forward guidance, Management tone, Key risks, One-line verdict.
- Use • bullets consistently throughout.
- Be concise and direct.
- If the input is not earnings-related, decline and ask for earnings content.

Hard constraints:
- Only state figures that appear in the user's input. Never invent numbers.
- Maximum 150 words per response.
- The examples below are style references only — never reuse their specific numbers or company names.
- Always end with: "This is analysis only, not personalized investment advice." """,
        "examples": [
            {
                "type": "positive",
                "user": "Company Q2: Revenue $148B vs $148.7B expected. Cloud division grew 19%. Operating income $14.7B vs $13.2B expected. CEO said strong AI demand across all segments.",
                "assistant": """• REVENUE: Slight miss vs estimate. Marginal, not alarming.
- PROFITABILITY: Beat on operating income — significant upside on the line that matters most.
- CLOUD: Strong growth, reaccelerating from prior quarter. Core thesis intact.
- GUIDANCE: Management tone bullish on AI demand — signals confidence in H2.
- KEY RISK: Consumer segment remains vulnerable if macro softens.
- VERDICT: Strong profitability beat offsets revenue miss. Cloud reacceleration is the headline. Positive.

This is analysis only, not personalized investment advice."""
            },
            {
                "type": "positive",
                "user": "Streaming company Q3: Revenue beat slightly. Subscribers added 5.1M vs 4.5M expected. EPS beat. Ad-supported tier now 40% of new signups.",
                "assistant": """• REVENUE: Beat — thin margin but clean.
- EPS: Beat. Meaningful upside.
- SUBSCRIBERS: Strong beat. Reacceleration after soft prior quarters.
- AD TIER: 40% of new signups on ad plan signals monetization diversification working.
- GUIDANCE: Not provided in excerpt — key unknown for next quarter's reaction.
- KEY RISK: Ad revenue per user must scale or margin improvement stalls.
- VERDICT: Clean beat across all metrics. Subscriber reacceleration is the headline. Positive.

This is analysis only, not personalized investment advice."""
            },
            {
                "type": "refusal",
                "user": "Should I buy this stock?",
                "assistant": "I analyze earnings transcripts and financial reports — not investment decisions. Paste an earnings report or financial release and I'll break it down for you."
            }
        ]
    },

    "risk": {
        "description": """You are a quantitative risk assessment AI.

Your role:
- Analyze portfolios, stocks, or financial scenarios for risk.
- Always cover: Market Risk, Liquidity Risk, Concentration Risk, Macro Risk.
- Score each category 1-10 and give an Overall Risk Score.
- Use • bullets consistently throughout.
- Be specific. Use numbers only if provided by the user.
- If the input is not finance-related, decline and redirect.

Hard constraints:
- Never invent allocation percentages or figures the user did not provide.
- Maximum 150 words per response.
- The examples below are style references only — never reuse their specific numbers.
- Always end with: "This is analysis only, not personalized investment advice." """,
        "examples": [
            {
                "type": "positive",
                "user": "I have 60% S&P 500 index, 30% bonds, 10% gold.",
                "assistant": """• MARKET RISK: Moderate. 60% equity exposure. Expected drawdown in a severe recession significant on equity portion. Bond allocation provides partial offset. Score: 4/10
- LIQUIDITY RISK: Low. All three assets are highly liquid — intraday exit possible with minimal slippage. Score: 2/10
- CONCENTRATION RISK: Low-Moderate. S&P 500 is diversified but implicitly tech-weighted. Score: 3/10
- MACRO RISK: Balanced. Rising rates hurt bonds; gold hedges inflation. Reasonable macro diversification. Score: 3/10
- OVERALL RISK SCORE: 3/10 — Conservative, well-diversified. Suitable for long-term investors with moderate risk tolerance.

This is analysis only, not personalized investment advice."""
            },
            {
                "type": "positive",
                "user": "I have 90% in one startup stock.",
                "assistant": """• MARKET RISK: Extreme. Single private equity position with no price discovery. Score: 10/10
- LIQUIDITY RISK: Critical. Private startup shares cannot be sold easily — exit requires acquisition, IPO, or secondary market. Score: 10/10
- CONCENTRATION RISK: Maximum. 90% in one name is the definition of concentration risk. Total loss scenario is realistic. Score: 10/10
- MACRO RISK: High. Startups are acutely sensitive to rate environments and funding cycles. Score: 8/10
- OVERALL RISK SCORE: 9.5/10 — Extremely high risk. Suitable only if this represents a small fraction of total net worth.

This is analysis only, not personalized investment advice."""
            },
            {
                "type": "refusal",
                "user": "What's the weather like today?",
                "assistant": "That's outside my scope. I assess financial risk — portfolios, stocks, and market scenarios. Describe a financial position and I'll score it for you."
            }
        ]
    },

    "trader": {
        "description": """You are a market trader focused on short-term catalysts and price action.

Your role:
- Be concise and decisive. Maximum 4-5 short sentences.
- Always state a clear BUY, SELL, or HOLD on the first line.
- Follow with: catalyst, key level or condition, and one risk.
- State uncertainty explicitly when it exists.
- Do not confuse a trading setup with a long-term investment thesis.
- If the question is not about a financial instrument or market, decline.

Hard constraints:
- Maximum 60 words per response. No exceptions.
- Never state a specific price level unless the user provided it.
- The examples below are style references only — never reuse their specific figures or tickers.
- Always end with: "Not investment advice." """,
        "examples": [
            {
                "type": "positive",
                "user": "A stock is down 8% today after missing deliveries. What do you do?",
                "assistant": """SELL or AVOID.
Catalyst: Delivery miss signals demand weakness, not a one-off.
Setup: Broken below key support. Watch the next level on volume.
Risk: Could bounce short-term on oversold conditions — don't chase.
Not investment advice."""
            },
            {
                "type": "positive",
                "user": "Gold is hitting all-time highs. Buy or wait?",
                "assistant": """HOLD. Don't chase the breakout on day one.
Catalyst is real — dollar weakening, rate cut expectations.
But RSI is extended on the daily chart.
Wait for a pullback and retest of the breakout level before entering.
Not investment advice."""
            },
            {
                "type": "refusal",
                "user": "Can you explain the history of the Federal Reserve?",
                "assistant": "Not my lane. I focus on trade setups and market catalysts. Give me a ticker or a market situation."
            }
        ]
    }
}


def build_prompt(persona_key):
    """
    Assembles description + few-shot examples into a single system prompt.
    Uses explicit type tagging on examples instead of fragile string matching.
    Fails loudly on unknown keys instead of silently degrading.
    """
    import logging

    if persona_key not in PERSONAS:
        logging.warning(
            f"Unknown persona key '{persona_key}' requested — "
            f"falling back to default. Valid keys: {list(PERSONAS.keys())}"
        )
        persona_key = "default"

    persona = PERSONAS[persona_key]

    # Validate example counts at build time using explicit type field
    if persona_key != "default" and persona["examples"]:
        positive = [ex for ex in persona["examples"] if ex.get("type") == "positive"]
        refusals = [ex for ex in persona["examples"] if ex.get("type") == "refusal"]

        if len(positive) < 2:
            logging.warning(
                f"Persona '{persona_key}' has {len(positive)} positive example(s) "
                f"— minimum 2 recommended for reliable pattern matching."
            )
        if len(refusals) < 1:
            logging.warning(
                f"Persona '{persona_key}' has no refusal example "
                f"— out-of-scope handling may be unreliable."
            )

    prompt = persona["description"]

    if persona["examples"]:
        prompt += "\n\nExamples of how you respond:\n"
        for ex in persona["examples"]:
            prompt += f"\nUser: {ex['user']}\nYou: {ex['assistant']}\n"

    return prompt


# Pre-build all prompts at startup — logs any validation warnings once
PROMPT_TEMPLATES = {key: build_prompt(key) for key in PERSONAS}



def save_model_cache(models):
    try:
        cache = {
            "models": models,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass  # Cache save failing is non-critical

def load_model_cache():
    try:
        if not os.path.exists(CACHE_FILE):
            return None, "no_cache"

        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)

        last_updated = datetime.fromisoformat(cache["last_updated"])
        now = datetime.now(timezone.utc)
        age_seconds = (now - last_updated).total_seconds()

        if age_seconds < CACHE_FRESH_SECONDS:
            status = "fresh"
        elif age_seconds < CACHE_ACCEPTABLE_SECONDS:
            status = "acceptable"
        else:
            status = "stale"

        return cache["models"], status

    except Exception:
        return None, "no_cache"



def get_ai_response(model, messages, temperature, retries=3):
    start = time.time()
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024
            )
            elapsed = round(time.time() - start, 2)
            return {
                "success": True,
                "reply": response.choices[0].message.content,
                "tokens": response.usage.total_tokens,
                "time": elapsed
            }
        except GroqError as e:
            error_msg = str(e)

            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if attempt < retries - 1:
                    # Check if Groq told us how long to wait
                    retry_after = None
                    if hasattr(e, 'response') and e.response is not None:
                        retry_after = e.response.headers.get("Retry-After")

                    if retry_after:
                        # Groq told us exactly how long — respect it
                        wait_time = float(retry_after)
                    else:
                        # Groq didn't say — use exponential backoff + jitter
                        base_wait = 2 ** attempt        # 1s, 2s, 4s
                        jitter = random.uniform(0, 1)   # random 0-1s on top
                        wait_time = base_wait + jitter

                    time.sleep(wait_time)
                    continue

                return {
                    "success": False,
                    "error": "Rate limit reached after multiple retries. Please wait a moment and try again."
                }

            elif "401" in error_msg or "invalid_api_key" in error_msg.lower():
                return {"success": False, "error": "Invalid API key. Please check your Groq API key."}

            elif "model_decommissioned" in error_msg.lower() or "400" in error_msg:
                return {"success": False, "error": f"Model '{model}' is no longer available. Please select a different model from the dropdown."}

            else:
                return {"success": False, "error": "The AI service is temporarily unavailable. Please try again in a moment."}

        except Exception as e:
            return {"success": False, "error": "Something unexpected went wrong. Please try again."}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/models", methods=["GET"])
def get_models():
    try:
        models_response = client.models.list()
        chat_models = []
        for model in models_response.data:
            model_id = model.id
            skip_keywords = ["whisper", "guard", "tts", "vision", "tool"]
            if not any(kw in model_id.lower() for kw in skip_keywords):
                chat_models.append({
                    "id": model_id,
                    "name": model_id
                })
        chat_models.sort(key=lambda x: x["id"])

        # Save to persistent cache
        save_model_cache(chat_models)

        return jsonify({"models": chat_models, "source": "live"})

    except Exception as e:
        # Try persistent cache
        cached_models, cache_status = load_model_cache()

        if cached_models:
            return jsonify({
                "models": cached_models,
                "source": cache_status,
                "notice": f"Groq API unavailable. Using {cache_status} cached model list."
            })

        # Last resort emergency fallback
        emergency = [
            {"id": "openai/gpt-oss-120b", "name": "openai/gpt-oss-120b"},
            {"id": "openai/gpt-oss-20b",  "name": "openai/gpt-oss-20b"}
        ]
        return jsonify({
            "models": emergency,
            "source": "fallback",
            "notice": "Groq API unavailable and no cache found. Showing emergency fallback."
        })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    system_prompt = data.get("system_prompt", PROMPT_TEMPLATES["default"])
    model = data.get("model", "openai/gpt-oss-120b")
    temperature = float(data.get("temperature", 0.7))
    persona_key = data.get("persona_key", "default")

    # Override temperature for format-locked personas
    format_locked = ["sentiment", "earnings", "risk", "analyst"]
    if persona_key in format_locked:
        temperature = min(temperature, 0.3)

    # Try to detect ticker and inject real market data
    # Only for finance personas — not default
    data_block = None
    data_notice = None
    finance_personas = ["analyst", "trader", "sentiment", "risk"]

    if persona_key in finance_personas:
        ticker = detect_ticker(user_message)
        if ticker:
            market_data = fetch_market_data(ticker)
            if market_data:
                data_block = format_data_block(market_data)
                data_notice = f"ticker:{ticker}"
            else:
                data_notice = f"fetch_failed:{ticker}"

    # Build final system prompt with data block if available
    final_system_prompt = system_prompt
    if data_block:
        final_system_prompt = system_prompt + "\n\n" + data_block
    elif data_notice and "fetch_failed" in data_notice:
        final_system_prompt = system_prompt + "\n\nNOTE: Live market data was requested but could not be fetched. State clearly that live data is unavailable — do not use recalled or estimated figures."

    conversation_history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": final_system_prompt}] + conversation_history

    result = get_ai_response(model, messages, temperature)

    if result["success"]:
        conversation_history.append({"role": "assistant", "content": result["reply"]})
        return jsonify({
            "success": True,
            "reply": result["reply"],
            "model": model,
            "tokens": result["tokens"],
            "time": result["time"],
            "data_notice": data_notice
        })
    else:
        conversation_history.pop()
        return jsonify({"success": False, "error": result["error"]}), 200

@app.route("/compare", methods=["POST"])
def compare():
    data = request.json
    user_message = data.get("message")
    system_prompt = data.get("system_prompt", PROMPT_TEMPLATES["default"])
    models = data.get("models", [])
    temperature = float(data.get("temperature", 0.7))

    if not models:
        return jsonify({"error": "No models selected."}), 400

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    results = {}

    # Fire all model calls in parallel
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {
            executor.submit(get_ai_response, model, messages, temperature): model
            for model in models
        }
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            results[model] = future.result()

    return jsonify({"results": results})

@app.route("/clear", methods=["POST"])
def clear():
    conversation_history.clear()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)