import json

import requests


def main() -> int:
    resp = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "model": "kingogpt-web",
            "messages": [
                {"role": "user", "content": "Say hello in one sentence."},
            ],
        },
        timeout=180,
    )

    print(resp.status_code)
    try:
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(resp.text)
    return 0 if resp.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
