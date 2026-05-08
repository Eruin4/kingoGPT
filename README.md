# kingoGPT

KingoGPT API/token automation utilities.

This repository contains the KingoGPT web API solver and token capture code used
by downstream services such as `hermingo` and `kingoAssistant`.

## Install

```bash
pip install git+https://github.com/Eruin4/kingoGPT.git
python -m playwright install chromium
```

## Commands

```bash
kingogpt-api-solver "hello"
kingogpt-token-capture --cache-file state/kingogpt_token_cache.json
```

Legacy script names are kept for compatibility:

```bash
python kingogpt_api_solver.py "hello"
python kingogpt_token_capture.py
```
