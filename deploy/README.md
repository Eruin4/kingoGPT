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
  -d '{"model":"internal-azure-web-agent","messages":[{"role":"user","content":"hello"}]}'
```

## Runtime State

`docker-compose.yml` mounts `./state` into `/app/state`, so token cache and profile data stay outside the image.
