import ollama

def generate_code(prompt: str):
    """
    Generates code using local Ollama model.
    """

    try:
        response = ollama.chat(
            model="nemotron-3-nano:30b-cloud",  # change if you use different model
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert software engineer. Generate clean, production-ready code only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Error generating code: {str(e)}"
