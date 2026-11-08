"""Check that the API keys in .env work, using read-only calls that spend no credits.

Run:  .venv/bin/python check_keys.py
"""

import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)  # .env wins over anything already set in the shell

# python.org's macOS Python ships without root certificates; use certifi's bundle if present.
try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
ADAPTION_BASE = os.environ.get("ADAPTION_BASE_URL", "https://api.prod.adaptionlabs.ai/api/v1").rstrip("/")


def get_key(name: str) -> str | None:
    key = (os.environ.get(name) or "").strip().strip("'\"")
    if not key:
        print(f"  ❌ {name} is empty. Paste it into .env right after the '='.")
        return None
    return key


@dataclass
class Response:
    status_code: int
    text: str

    def json(self) -> dict:
        return json.loads(self.text or "{}")


def get(url: str, key: str, **params) -> Response | None:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": "check_keys/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20, context=SSL_CONTEXT) as r:
            return Response(r.status, r.read().decode())
    except urllib.error.HTTPError as e:
        return Response(e.code, e.read().decode(errors="replace"))
    except urllib.error.URLError as e:
        print(f"  ❌ Couldn't reach {url}: {e.reason}")
        return None


def rejected(r: Response) -> bool:
    if r.status_code == 200:
        return False
    hint = " (key is wrong, revoked, or has a typo)" if r.status_code in (401, 403) else ""
    print(f"  ❌ Rejected: HTTP {r.status_code}{hint}")
    print(f"     {r.text[:200]}")
    return True


def check_openrouter() -> bool:
    print("OpenRouter")
    key = get_key("OPENROUTER_API_KEY")
    if not key:
        return False

    r = get(f"{OPENROUTER_BASE}/key", key)
    if r is None or rejected(r):
        return False

    info = r.json().get("data", {})
    print("  ✅ Key works")
    if info.get("label"):
        print(f"     Label:          {info['label']}")
    print(f"     Used so far:    ${info.get('usage', 0):.4f}")
    remaining = info.get("limit_remaining")
    print(f"     Key limit left: {'no per-key cap' if remaining is None else f'${remaining:.2f}'}")

    # Account-wide balance; not every key type can read it, so it's optional.
    c = get(f"{OPENROUTER_BASE}/credits", key)
    if c is not None and c.status_code == 200:
        d = c.json().get("data", {})
        total, used = d.get("total_credits", 0), d.get("total_usage", 0)
        print(f"     Account credit: ${total - used:.2f} left of ${total:.2f}")
    return True


def check_adaption() -> bool:
    print("Adaption")
    key = get_key("ADAPTION_API_KEY")
    if not key:
        return False

    r = get(f"{ADAPTION_BASE}/datasets", key, limit=100)
    if r is None or rejected(r):
        return False

    body = r.json()
    count = len(body.get("datasets", []))
    more = "+" if body.get("next_cursor") else ""
    print("  ✅ Key works")
    print(f"     Datasets on your team: {count}{more}")
    return True


if __name__ == "__main__":
    results = [check_openrouter(), check_adaption()]
    print()
    print("All keys working." if all(results) else "Some keys need fixing (see above).")
    sys.exit(0 if all(results) else 1)
