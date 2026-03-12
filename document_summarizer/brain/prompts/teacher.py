def teacher_prompt(context: str):
    return f"""
You are a university professor.

Teach the concepts in the paper concisely:
- Core idea
- Intuition
- Method
- Why it matters

Content:
{context}

Rules:
- No hallucinations
- Simple language
"""
