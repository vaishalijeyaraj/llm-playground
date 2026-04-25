from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

user_message = "Explain what an LLM is in 3 simple sentences."

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)