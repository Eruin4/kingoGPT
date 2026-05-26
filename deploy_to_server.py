import os
import subprocess
import sys
import time

# 로컬 및 원격 경로 정의
local_base = r"C:\Users\ppggh\.antigravity-ide\kingoGPT"
remote_user_host = "eruin@192.168.0.3"
remote_kingogpt_base = "/home/eruin/kingoGPT"
remote_hermingo_base = "/home/eruin/hermingo"

# 전송할 파일 목록 (로컬 상대경로, 원격 절대경로)
files_to_deploy = [
    (r"kingogpt\shared.py", f"{remote_kingogpt_base}/kingogpt/shared.py"),
    (r"kingogpt\exceptions.py", f"{remote_kingogpt_base}/kingogpt/exceptions.py"),
    (r"kingogpt\api_solver.py", f"{remote_kingogpt_base}/kingogpt/api_solver.py"),
    (r"kingogpt\token_capture.py", f"{remote_kingogpt_base}/kingogpt/token_capture.py"),
    (r"kingogpt\client.py", f"{remote_kingogpt_base}/kingogpt/client.py"),
    (r"tests\test_tool_adapter.py", f"{remote_kingogpt_base}/tests/test_tool_adapter.py"),
    (r"tests\test_api_solver_parsing.py", f"{remote_kingogpt_base}/tests/test_api_solver_parsing.py"),
    (r"hermingo\hermingo\server\openai_compat.py", f"{remote_hermingo_base}/hermingo/server/openai_compat.py"),
]

def run_local_cmd(cmd, cwd=r"C:\Windows", env=None):
    """로컬에서 명령어 실행 (getcwd() 에러 방지를 위해 C:\\Windows를 cwd로 지정)"""
    r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    return r

def run_remote_cmd(cmd_str):
    """원격 서버에서 SSH 명령 실행"""
    cmd = ['ssh', '-o', 'BatchMode=yes', remote_user_host, cmd_str]
    return run_local_cmd(cmd)

print("=== [1/4] GIT COMMIT & PUSH ===")
git_env = os.environ.copy()
git_env["GIT_DIR"] = os.path.join(local_base, ".git")
git_env["GIT_WORK_TREE"] = local_base

# git add
print("Adding files to git...")
run_local_cmd([r"C:\Program Files\Git\cmd\git.exe", "add", "."], env=git_env)

# git commit (이미 커밋되어 있으면 그냥 넘어감)
print("Committing files...")
r = run_local_cmd([r"C:\Program Files\Git\cmd\git.exe", "commit", "-m", "Improve KingoGPT client parity, structure logging, and typed exceptions"], env=git_env)

# git pull --rebase (충돌 해소)
print("Pulling remote changes (rebase)...")
r = run_local_cmd([r"C:\Program Files\Git\cmd\git.exe", "pull", "--rebase", "origin", "main"], env=git_env)
print("Pull stdout:")
print(r.stdout)
print("Pull stderr:")
print(r.stderr)

# git push
print("Pushing to GitHub...")
r = run_local_cmd([r"C:\Program Files\Git\cmd\git.exe", "push", "origin", "main"], env=git_env)
print("Push stdout:")
print(r.stdout)
print("Push stderr:")
print(r.stderr)


print("\n=== [2/4] SCP DEPLOYING FILES TO REMOTE SERVER ===")
for local_rel, remote_abs in files_to_deploy:
    local_abs = os.path.join(local_base, local_rel)
    print(f"Deploying: {local_rel} -> {remote_abs}")
    scp_cmd = ['scp', '-o', 'BatchMode=yes', local_abs, f"{remote_user_host}:{remote_abs}"]
    r = run_local_cmd(scp_cmd)
    if r.returncode != 0:
        print(f"Failed to deploy {local_rel}: {r.stderr}")
    else:
        print(f"Successfully deployed {local_rel}")


print("\n=== [3/4] RESTARTING DOCKER CONTAINER ON REMOTE ===")
print("Restarting hermingo docker container...")
r = run_remote_cmd(f"cd {remote_hermingo_base} && docker compose restart hermingo")
print("STDOUT:")
print(r.stdout)
print("STDERR:")
print(r.stderr)


print("\n=== [4/4] HEALTH CHECK ON REMOTE SERVER ===")
print("Waiting 5 seconds for uvicorn to spin up...")
time.sleep(5) # 컨테이너 내부 uvicorn 시작 대기

# 컨테이너 포트 헬스체크
r = run_remote_cmd("curl -s http://127.0.0.1:8000/health || curl -s http://127.0.0.1:8000/")
print("Health check response:")
print(r.stdout or r.stderr or "No response (or empty)")

print("\n=== DEPLOYMENT PROCESS COMPLETED ===")
