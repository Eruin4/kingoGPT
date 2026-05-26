import subprocess

cmd = ['ssh', '-o', 'BatchMode=yes', 'eruin@192.168.0.3', 'ls -la /home/eruin/kingoGPT']
r = subprocess.run(cmd, cwd=r"C:\Windows", capture_output=True, text=True, encoding='utf-8', errors='ignore')
print("STDOUT:")
print(r.stdout)
print("STDERR:")
print(r.stderr)
