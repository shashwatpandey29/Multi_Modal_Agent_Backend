from agents.llm_provider import chat_completion
from document_summarizer.brain.llm.base import BaseLLM


class ProviderLLM(BaseLLM):
    def __init__(self, use_case: str = "chat", system_prompt: str | None = None):
        self.use_case = use_case
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": prompt})

        return chat_completion(messages=messages, use_case=self.use_case)
