from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Separate history for main chat and comparison
conversation_history = []

MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.1-8b-instant": "Llama 3.1 8B (Fast)",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "openai/gpt-oss-20b": "GPT-OSS 20B"
}

PROMPT_TEMPLATES = {
    "default": "You are a helpful assistant.",
    "analyst": "You are a senior Wall Street financial analyst with 20 years experience. You speak in sharp, confident, data-driven sentences. You always mention risks alongside opportunities.",
    "sentiment": "You are a financial sentiment analysis engine. When given any financial text, news, or earnings data, you respond with: SENTIMENT (Bullish/Bearish/Neutral), CONFIDENCE (0-100%), KEY SIGNALS (bullet points), and TRADING IMPLICATION. Be precise and clinical.",
    "earnings": "You are an expert at analyzing earnings call transcripts. Extract and summarize: Revenue highlights, EPS vs expectations, Forward guidance, Management tone, Key risks mentioned, and One-line verdict. Use bullet points.",
    "risk": "You are a quantitative risk assessment AI. When given a portfolio, stock, or financial scenario, analyze: Market Risk, Liquidity Risk, Concentration Risk, Macro Risk, and give an overall Risk Score (1-10). Be specific and quantitative.",
    "trader": "You are an aggressive Wall Street trader. Short sentences. High conviction. No fluff. Always give a clear BUY, SELL, or HOLD with a one-line reason."
}

def get_ai_response(model, messages, temperature):
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024
    )
    elapsed = round(time.time() - start, 2)
    return {
        "reply": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "time": elapsed
    }

@app.route("/")
def home():
    return render_template("index.html", models=MODELS, templates=PROMPT_TEMPLATES)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    system_prompt = data.get("system_prompt", PROMPT_TEMPLATES["default"])
    model = data.get("model", "llama-3.3-70b-versatile")
    temperature = float(data.get("temperature", 0.7))

    conversation_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    result = get_ai_response(model, messages, temperature)

    conversation_history.append({"role": "assistant", "content": result["reply"]})

    return jsonify({
        "reply": result["reply"],
        "model": MODELS.get(model, model),
        "tokens": result["tokens"],
        "time": result["time"]
    })

@app.route("/compare", methods=["POST"])
def compare():
    data = request.json
    user_message = data.get("message")
    system_prompt = data.get("system_prompt", PROMPT_TEMPLATES["default"])
    model_a = data.get("model_a", "llama-3.3-70b-versatile")
    model_b = data.get("model_b", "llama-3.1-8b-instant")
    temperature = float(data.get("temperature", 0.7))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    result_a = get_ai_response(model_a, messages, temperature)
    result_b = get_ai_response(model_b, messages, temperature)

    return jsonify({
        "model_a": {"name": MODELS.get(model_a, model_a), "reply": result_a["reply"], "tokens": result_a["tokens"], "time": result_a["time"]},
        "model_b": {"name": MODELS.get(model_b, model_b), "reply": result_b["reply"], "tokens": result_b["tokens"], "time": result_b["time"]}
    })

@app.route("/clear", methods=["POST"])
def clear():
    conversation_history.clear()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)