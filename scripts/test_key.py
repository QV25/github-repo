from dotenv import load_dotenv
load_dotenv()                          # pakt .env in projectroot

import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")
print("Key begins with:", openai.api_key[:8], "…")

r = openai.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[{"role": "user", "content": "Ping"}],
    max_tokens=1,
)
print("✅ GPT response:", r.choices[0].message.content)

