import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "/mnt/f/wsl_share/models/HY-MT1.5-1.8B",
        "prompt": "翻译成英文：今天天气很好。",
        "max_tokens": 50
    }
)
print(response.json()["choices"][0]["text"])