import os
import re
from typing import Any, Dict, Iterator, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI
import concurrent.futures
import time

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
    client = OpenAI(**_resolve_openai_client_kwargs(model))

    if model.startswith(("openai/gpt-oss-", "gpt-oss-")):
        response = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
        )
        return (response.output_text or "").strip()

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


def _resolve_openai_client_kwargs(model: str) -> Dict[str, str]:
    nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    nvidia_base_url = os.getenv("NVIDIA_API_BASE_URL", "").strip()

    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is required for answer generation")

    if model.startswith(("openai/gpt-oss-", "gpt-oss-")):
        api_key = nvidia_api_key
        base_url = nvidia_base_url or "https://integrate.api.nvidia.com/v1"
    else:
        api_key = nvidia_api_key
        base_url = nvidia_base_url or "https://integrate.api.nvidia.com/v1"

    return {"api_key": api_key, "base_url": base_url}


def _stream_text_chunks(text: str) -> Iterator[str]:
    for chunk in re.findall(r"\S+\s*", text):
        if chunk:
            yield chunk


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _extract_block(prompt: str, start_marker: str, end_markers: List[str]) -> str:
    lower_prompt = prompt.lower()
    start_idx = lower_prompt.find(start_marker.lower())
    if start_idx < 0:
        return ""

    start_idx += len(start_marker)
    remainder = prompt[start_idx:]
    lower_remainder = remainder.lower()

    end_idx = len(remainder)
    for marker in end_markers:
        candidate = lower_remainder.find(marker.lower())
        if candidate >= 0:
            end_idx = min(end_idx, candidate)

    return remainder[:end_idx].strip()


def _extract_prompt_text(messages: List[Message]) -> str:
    return "\n\n".join(message.get("content", "") for message in messages if message.get("content"))


def _extract_relevant_sentences(context: str, question: str, limit: int = 4) -> List[str]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
        "how", "i", "in", "is", "it", "of", "on", "or", "paper", "that", "the",
        "their", "this", "to", "was", "what", "when", "where", "which", "who", "why",
        "with", "would", "you",
    }

    question_words = {
        word
        for word in re.findall(r"[A-Za-z0-9]+", (question or "").lower())
        if word not in stop_words and len(word) > 2
    }

    sentences = [
        _normalize_whitespace(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", context or "")
        if _normalize_whitespace(sentence)
    ]

    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        tokens = set(re.findall(r"[A-Za-z0-9]+", sentence.lower()))
        score = len(tokens & question_words)
        if score:
            scored.append((score, sentence))

    scored.sort(key=lambda item: (-item[0], len(item[1])))
    if scored:
        return [sentence for _, sentence in scored[:limit]]

    if sentences:
        return sentences[:limit]

    return []


def _offline_completion(messages: List[Message], use_case: str) -> str:
    prompt_text = _extract_prompt_text(messages)
    if not prompt_text:
        return "The configured LLM provider is unavailable right now."

    if use_case == "chat":
        context = _extract_block(prompt_text, "Paper Content:", ["Question:"])
        question = _extract_block(prompt_text, "Question:", [])
        relevant = _extract_relevant_sentences(context, question)

        if relevant:
            bullets = "\n".join(f"- {sentence}" for sentence in relevant)
            return (
                "The configured LLM provider is unavailable, so this answer uses only the paper text:\n"
                f"{bullets}"
            )

        return "The paper does not explicitly address this."

    if use_case == "analysis":
        context = _extract_block(prompt_text, "PAPER CONTENT:", [])
        relevant = _extract_relevant_sentences(context, "analysis")
        bullets = "\n".join(f"- {sentence}" for sentence in relevant)
        return (
            "=== SUMMARY ===\n"
            "The configured LLM provider is unavailable.\n\n"
            "=== KEY LEARNINGS ===\n"
            f"{bullets or '- Not explicitly stated in the provided content.'}\n\n"
            "=== MAIN CONTRIBUTIONS ===\n"
            "- Not explicitly stated in the provided content.\n\n"
            "=== LIMITATIONS ===\n"
            "- Not explicitly stated in the provided content."
        )

    if use_case == "code":
        return (
            "The configured LLM provider is unavailable. "
            "Please set the provider credentials in production and retry this request."
        )

    return "The configured LLM provider is unavailable right now."


def _openai_chat_completion_stream(messages: List[Message], model: str, temperature: float) -> Iterator[str]:
    client = OpenAI(**_resolve_openai_client_kwargs(model))

    if model.startswith(("openai/gpt-oss-", "gpt-oss-")):
        response = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
        )

        yield from _stream_text_chunks((response.output_text or "").strip())
        return

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
        openai_model = os.getenv("OPENAI_CODER_MODEL", os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b"))
        gemini_model = os.getenv("GEMINI_CODER_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        openrouter_model = os.getenv(
            "OPENROUTER_CODER_MODEL",
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )
        ollama_model = os.getenv("CODER_MODEL", "codellama:latest")
    elif use_case == "analysis":
        openai_model = os.getenv(
            "OPENAI_ANALYSIS_MODEL",
            os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")),
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
        openai_model = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b"))
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
    temperature = _env_float("LLM_TEMPERATURE", 0.3)
    timeout_sec = _env_int("LLM_TIMEOUT_SEC", 60)
    selected_model = (model_override or "").strip() or None
    models = _resolve_models_for_use_case(use_case)

    def _call_provider():
        return _openai_chat_completion(
            messages=messages,
            model=selected_model or models["openai"],
            temperature=temperature,
        )

    # run provider call with timeout in thread to avoid blocking indefinitely
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call_provider)
        try:
            start = time.perf_counter()
            return fut.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise RuntimeError(f"LLM provider call timed out after {timeout_sec}s")
        except Exception:
            return _offline_completion(messages=messages, use_case=use_case)


def chat_completion_stream(
    messages: List[Message],
    use_case: str = "chat",
    model_override: Optional[str] = None,
) -> Iterator[str]:
    temperature = _env_float("LLM_TEMPERATURE", 0.3)
    timeout_sec = _env_int("LLM_TIMEOUT_SEC", 60)
    selected_model = (model_override or "").strip() or None
    models = _resolve_models_for_use_case(use_case)

    try:
        return _openai_chat_completion_stream(
            messages=messages,
            model=selected_model or models["openai"],
            temperature=temperature,
        )
    except Exception:
        return _stream_text_chunks(_offline_completion(messages=messages, use_case=use_case))