# ⚡ LLM Playground

A full-stack AI playground for experimenting with large language models in real time — built with a focus on financial analysis applications.

**Live Demo:** https://llmplayground.up.railway.app/  
**Built by:** Your Name Here  ← change this

---

## What This Is

LLM Playground is an interactive web application that lets you chat with,
compare, and stress-test multiple large language models simultaneously.
It was built as a deep-dive into how LLMs work — from tokenization and
temperature sampling to prompt engineering and model comparison.

---

## Features

- **Multi-Model Chat** — Switch between Llama 3.3 70B, Llama 3.1 8B,
  GPT-OSS 120B, and GPT-OSS 20B in real time
- **Side-by-Side Model Comparison** — Send one prompt to two models
  simultaneously and compare quality, speed, and token usage
- **Temperature Control** — Adjust model creativity live and observe
  output variance
- **Finance Prompt Templates** — One-click modes for:
  - Financial Analyst
  - Sentiment Analysis (Bullish/Bearish/Neutral with confidence score)
  - Earnings Call Summarizer
  - Quantitative Risk Assessment (scores Market, Liquidity,
    Concentration, and Macro risk 1-10)
  - Trader Mode
- **Token + Latency Tracking** — Every response shows token count
  and response time in milliseconds
- **Chat Export** — Download full conversations as .txt files
- **Conversation Memory** — Full context window maintained across
  the session

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Vanilla JavaScript |
| Backend | Python, Flask |
| LLM API | Groq (free tier) |
| Models | Llama 3.3 70B, Llama 3.1 8B, GPT-OSS 120B, GPT-OSS 20B |

---

## Key Observations from Building This

- **Model guardrails differ significantly** — GPT-OSS 120B refused
  to give direct investment advice while Llama 3.3 70B engaged freely.
  This has direct implications for compliance in fintech applications.
- **Temperature has nonlinear effects** — At temp=0, models are
  deterministic and clinical. At temp=1, outputs become creative but
  occasionally hallucinate financial data.
- **Token efficiency varies** — Llama 3.1 8B answered in ~250 tokens
  what GPT-OSS 120B took 1100+ tokens for, with comparable accuracy
  on structured tasks. Speed vs depth tradeoff is real.
- **System prompts are the real leverage** — The same base model
  behaves like a hedge fund analyst or a casual assistant purely
  based on the system prompt. Prompt engineering is a first-class skill.

---

## How to Run Locally

1. Clone the repo
```bash
   git clone https://github.com/yourusername/llm-playground.git
   cd llm-playground
```

2. Create virtual environment
```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Add your API key
```bash
   # Create a .env file
   GROQ_API_KEY=your_groq_key_here
```
   Get a free key at groq.com

5. Run
```bash
   python app.py
```
   Open http://127.0.0.1:5000

---

## Why I Built This

I built this to develop hands-on intuition for how LLMs work beyond
just using them as tools. Understanding the difference between models,
how temperature affects outputs, how system prompts shape behavior,
and how token economics work is essential for building serious
AI-powered financial applications.

---

## What's Next

- [ ] Streaming responses (token by token output)
- [ ] RAG integration (upload a PDF earnings report, chat with it)
- [ ] Fine-tuning experiments on financial data
- [ ] Prompt injection attack demonstrations