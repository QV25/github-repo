import os, time, random
import openai
from dotenv import load_dotenv
from openai.error import (Timeout, RateLimitError,
                          APIError, APIConnectionError)

load_dotenv()                              # leest .env
openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt(content: str,
        model="gpt-3.5-turbo-0125",
        temperature: float = 0.0,
        max_tokens: int = 50,
        retries: int = 4,
        timeout: int = 60) -> str:
    """
    Eén simpele GPT-wrapper met retries en exponential back-off.
    """
    for attempt in range(retries):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,          # <<< belangrijk
            )
            return resp.choices[0].message.content.strip()

        except (Timeout, APIConnectionError):
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"[timeout] retry {attempt+1}/{retries} in {wait:.1f}s")
            time.sleep(wait)

        except RateLimitError:
            wait = 10 + 5 * attempt       # wat langer bij rate-limit
            print(f"[rate-limit] retry {attempt+1}/{retries} in {wait}s")
            time.sleep(wait)

        except APIError as e:
            if 500 <= e.status_code < 600:        # tijdelijke 5xx
                wait = 2 ** attempt
                print(f"[5xx] retry {attempt+1}/{retries} in {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("OpenAI-call bleef mislukken na retries")


