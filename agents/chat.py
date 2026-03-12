import ollama
async def generate_text(prompt: str):
    try:
        response = ollama.chat(
            model="nemotron-3-nano:30b-cloud",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "status": "success",
            "response": response["message"]["content"]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
