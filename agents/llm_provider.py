import os
import re
from typing import Any, Dict, Iterator, List, Optional

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


def _parse_cost(value: Any) -> float:
    if value is None:
        return 1.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return 1.0

        try:
            return float(raw)
        except ValueError:
            return 1.0

    return 1.0


def _is_openrouter_free_model(model_data: Dict[str, Any]) -> bool:
    model_id = str(model_data.get("id", "")).strip().lower()
    if not model_id:
        return False

    if model_id.endswith(":free"):
        return True

    pricing = model_data.get("pricing", {})
    if not isinstance(pricing, dict):
        return False

    prompt_cost = _parse_cost(pricing.get("prompt"))
    completion_cost = _parse_cost(pricing.get("completion"))
    return prompt_cost <= 0 and completion_cost <= 0


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
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "ollama package is not installed. Install ollama or switch LLM_PROVIDER to openai/gemini/openrouter"
        ) from exc

    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"].strip()


def _stream_text_chunks(text: str) -> Iterator[str]:
    for chunk in re.findall(r"\S+\s*", text):
        if chunk:
            yield chunk


def _openai_chat_completion_stream(messages: List[Message], model: str, temperature: float) -> Iterator[str]:
    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    for part in stream:
        if not part.choices:
            continue

        delta = part.choices[0].delta
        content = delta.content if delta else None
        if content:
            yield content


def _openrouter_chat_completion_stream(messages: List[Message], model: str, temperature: float) -> Iterator[str]:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_require_env("OPENROUTER_API_KEY"),
    )
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    for part in stream:
        if not part.choices:
            continue

        delta = part.choices[0].delta
        content = delta.content if delta else None
        if content:
            yield content


def _gemini_chat_completion_stream(
    messages: List[Message],
    model: str,
    temperature: float,
    timeout_sec: int,
) -> Iterator[str]:
    text = _gemini_chat_completion(messages=messages, model=model, temperature=temperature, timeout_sec=timeout_sec)
    yield from _stream_text_chunks(text)


def _ollama_chat_completion_stream(messages: List[Message], model: str) -> Iterator[str]:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "ollama package is not installed. Install ollama or switch LLM_PROVIDER to openai/gemini/openrouter"
        ) from exc

    response = ollama.chat(model=model, messages=messages, stream=True)
    for part in response:
        content = str(part.get("message", {}).get("content", ""))
        if content:
            yield content


def get_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def list_openrouter_free_models() -> List[Dict[str, Any]]:
    api_key = _require_env("OPENROUTER_API_KEY")
    timeout_sec = _env_int("LLM_TIMEOUT_SEC", 60)

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
    }

    referer = os.getenv("OPENROUTER_SITE_URL", "").strip()
    title = os.getenv("OPENROUTER_APP_NAME", "NEXUS")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=timeout_sec,
    )
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter models API error ({response.status_code}): {response.text}")

    payload = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []

    free_models: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        if not _is_openrouter_free_model(item):
            continue

        model_id = str(item.get("id", "")).strip()
        if not model_id:
            continue

        name = str(item.get("name") or model_id).strip()
        model_payload: Dict[str, Any] = {
            "id": model_id,
            "name": name,
        }

        context_length = item.get("context_length")
        if isinstance(context_length, int):
            model_payload["context_length"] = context_length

        free_models.append(model_payload)

    free_models.sort(key=lambda model: str(model.get("name", "")).lower())
    return free_models


def _resolve_models_for_use_case(use_case: str) -> Dict[str, str]:
    if use_case == "code":
        openai_model = os.getenv("OPENAI_CODER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        gemini_model = os.getenv("GEMINI_CODER_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        openrouter_model = os.getenv(
            "OPENROUTER_CODER_MODEL",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )
        ollama_model = os.getenv("CODER_MODEL", "codellama:latest")
    elif use_case == "analysis":
        openai_model = os.getenv(
            "OPENAI_ANALYSIS_MODEL",
            os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        )
        gemini_model = os.getenv(
            "GEMINI_ANALYSIS_MODEL",
            os.getenv("GEMINI_CHAT_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")),
        )
        openrouter_model = os.getenv(
            "OPENROUTER_ANALYSIS_MODEL",
            os.getenv("OPENROUTER_CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")),
        )
        ollama_model = os.getenv("ANALYSIS_MODEL", os.getenv("SUMMARIZER_MODEL", "llama3:latest"))
    else:
        openai_model = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        gemini_model = os.getenv("GEMINI_CHAT_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        openrouter_model = os.getenv(
            "OPENROUTER_CHAT_MODEL",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )
        ollama_model = os.getenv("SUMMARIZER_MODEL", "llama3:latest")

    return {
        "openai": openai_model,
        "gemini": gemini_model,
        "openrouter": openrouter_model,
        "ollama": ollama_model,
    }


def chat_completion(
    messages: List[Message],
    use_case: str = "chat",
    model_override: Optional[str] = None,
) -> str:
    provider = get_llm_provider()
    temperature = _env_float("LLM_TEMPERATURE", 0.3)
    timeout_sec = _env_int("LLM_TIMEOUT_SEC", 60)
    selected_model = (model_override or "").strip() or None
    models = _resolve_models_for_use_case(use_case)

    if provider in {"chatgpt", "openai"}:
        return _openai_chat_completion(
            messages=messages,
            model=selected_model or models["openai"],
            temperature=temperature,
        )

    if provider == "gemini":
        return _gemini_chat_completion(
            messages=messages,
            model=selected_model or models["gemini"],
            temperature=temperature,
            timeout_sec=timeout_sec,
        )

    if provider == "openrouter":
        return _openrouter_chat_completion(
            messages=messages,
            model=selected_model or models["openrouter"],
            temperature=temperature,
        )

    if provider == "ollama":
        return _ollama_chat_completion(messages=messages, model=selected_model or models["ollama"])

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use one of: chatgpt, openai, gemini, openrouter, ollama"
    )


def chat_completion_stream(
    messages: List[Message],
    use_case: str = "chat",
    model_override: Optional[str] = None,
) -> Iterator[str]:
    provider = get_llm_provider()
    temperature = _env_float("LLM_TEMPERATURE", 0.3)
    timeout_sec = _env_int("LLM_TIMEOUT_SEC", 60)
    selected_model = (model_override or "").strip() or None
    models = _resolve_models_for_use_case(use_case)

    if provider in {"chatgpt", "openai"}:
        return _openai_chat_completion_stream(
            messages=messages,
            model=selected_model or models["openai"],
            temperature=temperature,
        )

    if provider == "gemini":
        return _gemini_chat_completion_stream(
            messages=messages,
            model=selected_model or models["gemini"],
            temperature=temperature,
            timeout_sec=timeout_sec,
        )

    if provider == "openrouter":
        return _openrouter_chat_completion_stream(
            messages=messages,
            model=selected_model or models["openrouter"],
            temperature=temperature,
        )

    if provider == "ollama":
        return _ollama_chat_completion_stream(messages=messages, model=selected_model or models["ollama"])

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use one of: chatgpt, openai, gemini, openrouter, ollama"
    )