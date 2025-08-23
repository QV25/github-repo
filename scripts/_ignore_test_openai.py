import os, openai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())          # leest ~/thesis-project/.env

openai.api_key = os.getenv("OPENAI_API_KEY")

r = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say hi briefly"}],
    max_tokens=5,
    temperature=0,
)
print(r.choices[0].message.content)

