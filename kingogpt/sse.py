"""KingoGPT SSE framing and event validation (web build deznAsajYjJTwd-t8jhRP)."""
import json

from kingogpt.exceptions import AuthenticationError, BackendError


def iter_events(lines, *, max_event_chars=2_000_000):
    """Decode UTF-8 SSE data fields, including multiline data and heartbeats."""
    fields, size, event_type = [], 0, ""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8-sig")
        line = line.removesuffix("\r")
        if not line:
            if fields:
                yield event_type, "\n".join(fields)
            fields, size, event_type = [], 0, ""
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event_type = value
        elif field == "data":
            fields.append(value)
            size += len(value) + 1
            if size > max_event_chars:
                raise BackendError("Upstream SSE event exceeds the size limit.")
    if fields:
        # Some proxies omit the final blank line when closing a completed stream.
        yield event_type, "\n".join(fields)


def document(event):
    data = event.get("data")
    documents = data.get("documents") if isinstance(data, dict) else None
    return documents[0] if isinstance(documents, list) and documents and isinstance(documents[0], dict) else {}


def decode_event(event_type, raw):
    try:
        event = json.loads(raw)
    except ValueError as exc:
        raise BackendError("Upstream sent invalid JSON in an SSE event.") from exc
    if not isinstance(event, dict):
        raise BackendError("Upstream SSE event must be a JSON object.")
    code = event.get("code")
    if str(code) == "00001":
        raise AuthenticationError("Upstream SSE reported an expired session (00001).")
    if event_type == "error" or event.get("error") or (code is not None and str(code) != "200"):
        # Do not expose arbitrary upstream response bodies containing account data.
        raise BackendError("Upstream reported a failed SSE event.")
    return event
