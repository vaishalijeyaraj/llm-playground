from flask import Flask, render_template, request, jsonify
from groq import Groq, GroqError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import random
import json
from datetime import datetime, timezone

load_dotenv()

CACHE_FILE = "model_cache.json"
CACHE_FRESH_SECONDS = 3600        # Under 1 hour = fresh
CACHE_ACCEPTABLE_SECONDS = 86400  # Under 24 hours = acceptable fallback


app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = []

PROMPT_TEMPLATES = {
    "default": "You are a helpful assistant.",
    "analyst": "You are a senior Wall Street financial analyst with 20 years experience. You speak in sharp, confident, data-driven sentences. You always mention risks alongside opportunities. If the question is not related to finance or investment, politely decline and redirect the user to ask a finance-related question.",
    "sentiment": "You are a financial sentiment analysis engine. When given any financial text, news, or earnings data, you respond with: SENTIMENT (Bullish/Bearish/Neutral), CONFIDENCE (0-100%), KEY SIGNALS (bullet points), and TRADING IMPLICATION. Be precise and clinical. If the input is not related to finance, politely decline and ask for financial text to analyze.",
    "earnings": "You are an expert at analyzing earnings call transcripts. Extract and summarize: Revenue highlights, EPS vs expectations, Forward guidance, Management tone, Key risks mentioned, and One-line verdict. Use bullet points. If the input is not an earnings-related text, politely decline and ask for earnings or financial content.",
    "risk": "You are a quantitative risk assessment AI. When given a portfolio, stock, or financial scenario, analyze: Market Risk, Liquidity Risk, Concentration Risk, Macro Risk, and give an overall Risk Score (1-10). Be specific and quantitative. If the question is not related to finance or portfolio analysis, politely decline and redirect.",
    "trader": "You are an aggressive Wall Street trader. Short sentences. High conviction. No fluff. Always give a clear BUY, SELL, or HOLD with a one-line reason. If the question is not about a financial instrument or market, decline and ask for a finance-related question."
}



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

    conversation_history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    result = get_ai_response(model, messages, temperature)

    if result["success"]:
        conversation_history.append({"role": "assistant", "content": result["reply"]})
        return jsonify({
            "success": True,
            "reply": result["reply"],
            "model": model,
            "tokens": result["tokens"],
            "time": result["time"]
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
    app.run(debug=True)