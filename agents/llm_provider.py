import os
from typing import Dict, List

import ollama
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


Message = Dict[str, str]


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value


def _env_float(var_name: str, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is None:
        return default

    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(var_name: str, default: int) -> int:
    raw = os.getenv(var_name)
    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError:
        return default


def _gemini_prompt_from_messages(messages: List[Message]) -> str:
    chunks = []
    for message in messages:
        role = message.get("role", "user").strip().lower()
        content = message.get("content", "").strip()
        if not content:
            continue

        if role == "system":
            label = "System"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = "User"

        chunks.append(f"{label}: {content}")

    return "\n\n".join(chunks)


def _openai_chat_completion(messages: List[Message], model: str, temperature: float) -> str:
    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    content = response.choices[0].message.content if response.choices else ""
    return (content or "").strip()


def _openrouter_chat_completion(messages: List[Message], model: str, temperature: float) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_require_env("OPENROUTER_API_KEY"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    content = response.choices[0].message.content if response.choices else ""
    return (content or "").strip()


def _gemini_chat_completion(messages: List[Message], model: str, temperature: float, timeout_sec: int) -> str:
    api_key = _require_env("GOOGLE_API_KEY")
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _gemini_prompt_from_messages(messages),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    response = requests.post(endpoint, json=payload, timeout=timeout_sec)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")

    return text


def _ollama_chat_completion(messages: List[Message], model: str) -> str:
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"].strip()


def chat_completion(messages: List[Message], use_case: str = "chat") -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    temperature = _env_float("LLM_TEMPERATURE", 0.3)
    timeout_sec = _env_int("LLM_TIMEOUT_SEC", 60)

    if use_case == "code":
        openai_model = os.getenv("OPENAI_CODER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        gemini_model = os.getenv("GEMINI_CODER_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        openrouter_model = os.getenv(
            "OPENROUTER_CODER_MODEL",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )
        ollama_model = os.getenv("CODER_MODEL", "codellama:latest")
    else:
        openai_model = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        gemini_model = os.getenv("GEMINI_CHAT_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        openrouter_model = os.getenv(
            "OPENROUTER_CHAT_MODEL",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )
        ollama_model = os.getenv("SUMMARIZER_MODEL", "llama3:latest")

    if provider in {"chatgpt", "openai"}:
        return _openai_chat_completion(messages=messages, model=openai_model, temperature=temperature)

    if provider == "gemini":
        return _gemini_chat_completion(
            messages=messages,
            model=gemini_model,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )

    if provider == "openrouter":
        return _openrouter_chat_completion(
            messages=messages,
            model=openrouter_model,
            temperature=temperature,
        )

    if provider == "ollama":
        return _ollama_chat_completion(messages=messages, model=ollama_model)

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use one of: chatgpt, openai, gemini, openrouter, ollama"
    )