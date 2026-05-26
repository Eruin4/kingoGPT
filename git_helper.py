import os
import subprocess
import sys

# 첫 번째 인자가 'hermingo' 이면 hermingo 리포지토리를 대상으로 삼고,
# 그렇지 않으면 kingoGPT 리포지토리를 기본 대상으로 삼습니다.
repo_type = "kingoGPT"
args = sys.argv[1:]

if args and args[0] in ("kingoGPT", "hermingo"):
    repo_type = args[0]
    args = args[1:]

if repo_type == "hermingo":
    repo_dir = r"C:\Users\ppggh\.antigravity-ide\kingoGPT\hermingo"
else:
    repo_dir = r"C:\Users\ppggh\.antigravity-ide\kingoGPT"

git_dir = os.path.join(repo_dir, ".git")

# 환경 변수 강제 설정
env = os.environ.copy()
env["GIT_DIR"] = git_dir
env["GIT_WORK_TREE"] = repo_dir

if not args:
    args = ["status"]

# git.exe의 절대 경로를 직접 사용
git_path = r"C:\Program Files\Git\cmd\git.exe"
cmd = [git_path] + args

# cwd를 C:\Windows 로 설정하여 getcwd() Permission 에러 및 디렉토리NotFound 방지!
r = subprocess.run(cmd, env=env, cwd=r"C:\Windows", capture_output=True, text=True)

sys.stdout.write(r.stdout)
sys.stderr.write(r.stderr)
sys.exit(r.returncode)
