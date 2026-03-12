def full_analysis_prompt(context: str) -> str:
    return f"""
You are a senior academic research analyst.

Your task is to analyze ONLY the provided research paper content and produce a structured academic analysis.

⚠️ STRICT RULES:
- Use ONLY the given content.
- Do NOT introduce external knowledge.
- Do NOT assume missing details.
- If something is not mentioned, write:
  "Not explicitly stated in the provided content."
- Maintain formal academic tone.
- Be concise but information-dense.
- Do NOT include any text outside the required format.

------------------------------------------------------
REQUIRED OUTPUT FORMAT (STRICTLY FOLLOW):

=== SUMMARY ===
Write a structured academic summary covering:
- Problem Statement
- Proposed Method
- Key Experiments / Results
Keep it clear and well organized in paragraphs.

=== KEY LEARNINGS ===
Provide bullet points of the most important insights derived from the paper.

=== MAIN CONTRIBUTIONS ===
Provide bullet points describing the novel contributions of the paper.

=== LIMITATIONS ===
Provide bullet points listing explicit limitations.
If none mentioned, write:
Not explicitly stated in the provided content.

------------------------------------------------------
PAPER CONTENT:
{context}
"""
