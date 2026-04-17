from agents.llm_provider import chat_completion


async def generate_text(prompt: str):
    try:
        response_text = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            use_case="chat",
        )

        return {
            "status": "success",
            "response": response_text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
