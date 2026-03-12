def ask_prompt(context: str, question: str) -> str:
    return f"""
You are an academic assistant.

Answer the user's question using ONLY the provided paper content.

Rules:
- Answer in 3–5 bullet points OR 3–4 sentences
- Be direct and concise
- Do NOT speculate
- If the paper does not address the question, say:
  "The paper does not explicitly address this."

Paper Content:
{context}

Question:
{question}
"""
