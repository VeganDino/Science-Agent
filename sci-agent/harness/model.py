from openai import OpenAI

_client = None
MODEL_ID = "model"
BASE_URL = "http://localhost:8000/v1"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key="none")
    return _client


def generate(prompt: str, system_prompt: str = "", max_tokens: int = 2048, temperature: float = 0.2) -> str:
    client = _get_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
