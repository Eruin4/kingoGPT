import contextlib
import io


def run_python(code: str) -> str:
    """
    MVP-only Python runner. Production use requires a real sandbox.
    """
    allowed_globals = {
        "__builtins__": {
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "range": range,
            "print": print,
        }
    }
    local_vars = {}
    stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, allowed_globals, local_vars)
        output = stdout.getvalue()
        if output:
            return f"stdout={output!r}\nlocals={local_vars}"
        return str(local_vars)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"

