# Server Deployment

This package is prepared to run the OpenAI-compatible API in Docker on `eruin@192.168.0.3`.

## Local Sync

From this repository on Windows:

```powershell
.\deploy\sync_to_server.ps1
```

That excludes `state/` and `secrets/`. To also copy the current KingoGPT token/config state:

```powershell
.\deploy\sync_to_server.ps1 -IncludeState
```

The script uses `ssh`/`scp` and will prompt for the SSH password if no key is configured.

## Server Start

On the server:

```bash
cd ~/kingoGPT
./deploy/remote_bootstrap.sh
```

The API listens on server-local `127.0.0.1:8000`.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

OpenAI-compatible call:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kingogpt-web","messages":[{"role":"user","content":"hello"}]}'
```

## Hermes Provider Notes

Point Hermes at this server as an OpenAI-compatible provider:

- base URL: `http://127.0.0.1:8000/v1`
- model: `kingogpt-web`
- API key: any non-empty placeholder, if Hermes requires one

`/v1/chat/completions` is the preferred path. `/v1/responses` is available for clients that choose the Responses API, but hosted OpenAI tools are accepted only for compatibility and are not executed by this server yet.

## Runtime State

`docker-compose.yml` mounts `./state` into `/app/state`, so token cache and profile data stay outside the image.
