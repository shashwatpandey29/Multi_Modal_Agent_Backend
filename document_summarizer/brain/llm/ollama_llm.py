import requests
from document_summarizer.brain.logger import get_logger
from document_summarizer.brain.exceptions import LLMError

logger = get_logger("OllamaLLM")

class OllamaLLM:
    def __init__(self, model: str):
        self.model = model
        self.base_url = "http://127.0.0.1:11434"

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 700
                    }
                }
            )

            if response.status_code != 200:
                logger.error(f"Ollama error response: {response.text}")
                raise LLMError("Ollama request failed")

            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.Timeout:
            logger.error("Ollama timed out")
            raise LLMError("LLM response timeout")

        except Exception as e:
            logger.error(f"Ollama failed: {e}")
            raise LLMError(str(e))
