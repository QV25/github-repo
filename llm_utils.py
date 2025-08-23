import os, time, random
from typing import Optional, Sequence
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APITimeoutError as Timeout,
    RateLimitError,
    APIError,
    APIConnectionError,
    BadRequestError,
)

load_dotenv()

_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)

FALLBACK_MODELS: Sequence[str] = ("gpt-4o-mini", "gpt-4.1-mini")

def _responses_create_smart(model: str, input_text: str,
                            temperature: Optional[float], max_tok: int, timeout: int):
    include_temp = (temperature is not None)
    while True:
        try:
            params = dict(model=model, input=input_text, max_output_tokens=max_tok, timeout=timeout)
            if include_temp:
                params["temperature"] = temperature
            return _client.responses.create(**params)
        except BadRequestError as e:
            msg = (str(e) or "").lower()
            if include_temp and "temperature" in msg and ("unsupported value" in msg or "unsupported parameter" in msg):
                include_temp = False
                continue
            raise

def _chat_create_smart(model: str, messages,
                       temperature: Optional[float], max_tok: int, timeout: int):
    last_exc = None
    for token_name in ("max_completion_tokens", "max_tokens"):
        include_temp = (temperature is not None)
        tried = set()
        while True:
            key = (token_name, include_temp)
            if key in tried:
                break
            tried.add(key)
            params = dict(model=model, messages=messages, timeout=timeout)
            params[token_name] = max_tok
            if include_temp:
                params["temperature"] = temperature
            try:
                return _client.chat.completions.create(**params)
            except BadRequestError as e:
                last_exc = e
                msg = (str(e) or "").lower()
                if include_temp and "temperature" in msg and ("unsupported value" in msg or "unsupported parameter" in msg):
                    include_temp = False
                    continue
                if token_name == "max_completion_tokens" and ("unsupported parameter" in msg or "unknown parameter" in msg):
                    break
                raise
    if last_exc:
        raise last_exc
    raise RuntimeError("unreachable")

def _extract_text_from_responses(r) -> str:
    # 1) preferred helper
    txt = (getattr(r, "output_text", None) or "").strip()
    if txt:
        return txt
    # 2) walk the low-level structure
    try:
        d = r.model_dump()
        out = d.get("output") or d.get("response", {}).get("output") or []
        parts = []
        for item in out:
            if not isinstance(item, dict): 
                continue
            for c in item.get("content", []):
                if isinstance(c, dict):
                    t = c.get("text") or {}
                    if isinstance(t, str):
                        parts.append(t)
                    elif isinstance(t, dict):
                        val = t.get("value")
                        if isinstance(val, str):
                            parts.append(val)
        return "\n".join(p for p in parts if p).strip()
    except Exception:
        return ""

def gpt(content: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 50,
        retries: int = 4,
        timeout: int = 60) -> str:
    """
    - gpt-5-* → Responses API; anders Chat Completions.
    - automatische token/temperature compat.
    - lege output → fallback naar gpt-4o-mini, dan gpt-4.1-mini.
    """
    primary = (model or os.getenv("CHAT_MODEL", "gpt-5-mini")).strip()
    candidates = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    last_err = None

    for m in candidates:
        use_responses = m.startswith("gpt-5")
        for attempt in range(retries):
            try:
                if use_responses:
                    r = _responses_create_smart(
                        model=m,
                        input_text=content,
                        temperature=temperature,
                        max_tok=max_tokens,
                        timeout=timeout,
                    )
                    text = _extract_text_from_responses(r)
                    if text:
                        return text
                    # fallback naar chat-pad één keer als responses leeg is
                    resp = _chat_create_smart(
                        model=m,
                        messages=[{"role": "user", "content": content}],
                        temperature=temperature,
                        max_tok=max_tokens,
                        timeout=timeout,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    if text:
                        return text
                    # leeg → breek de retry-loop en probeer volgend model
                    break
                else:
                    resp = _chat_create_smart(
                        model=m,
                        messages=[{"role": "user", "content": content}],
                        temperature=temperature,
                        max_tok=max_tokens,
                        timeout=timeout,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    if text:
                        return text
                    break  # leeg → probeer volgend model

            except (Timeout, APIConnectionError) as e:
                last_err = e
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"[timeout/conn:{m}] retry {attempt+1}/{retries} in {wait:.1f}s")
                time.sleep(wait)
            except RateLimitError as e:
                last_err = e
                wait = 10 + 5 * attempt
                print(f"[rate-limit:{m}] retry {attempt+1}/{retries} in {wait}s")
                time.sleep(wait)
            except APIError as e:
                last_err = e
                status = getattr(e, "status_code", None)
                if status and 500 <= status < 600:
                    wait = 2 ** attempt
                    print(f"[5xx:{m}] retry {attempt+1}/{retries} in {wait}s")
                    time.sleep(wait)
                else:
                    # niet-retrybare fout → probeer volgend model
                    break

    # als we hier zijn: alles leeg of fout → geef laatste fout door of return lege string
    if last_err:
        raise RuntimeError(f"OpenAI-call bleef mislukken of gaf lege output: {last_err}")
    return ""

