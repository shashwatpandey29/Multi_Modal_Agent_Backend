def summary_prompt(context: str) -> str:
    return f"""
You are an academic research assistant.

Task:
Create a structured summary of the research paper using ONLY the provided content.

Required format:
1. Problem Statement
2. Proposed Method
3. Key Experiments / Results
4. Main Contributions
5. Limitations (ONLY if explicitly mentioned)

Rules:
- Do NOT add external knowledge
- Do NOT guess missing details
- If something is not mentioned, write "Not explicitly stated"
- Use concise bullet points

Paper Content:
{context}
"""
