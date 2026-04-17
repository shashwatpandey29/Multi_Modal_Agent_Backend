from agents.llm_provider import chat_completion

def generate_code(prompt: str):
    """
    Generates code using the provider selected in .env (OpenAI/ChatGPT, Gemini, OpenRouter, or Ollama).
    """

    try:
        return chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert software engineer. Generate clean, production-ready code only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            use_case="code",
        )

    except Exception as e:
        return f"Error generating code: {str(e)}"
