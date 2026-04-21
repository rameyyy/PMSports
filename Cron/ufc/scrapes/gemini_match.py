import os
import json
import time
import unicodedata
import requests


def _ascii_safe(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", errors="replace").decode("ascii")


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS = [
    "gemini-2.0-flash-lite",  # Gemini 3.1 Flash Lite — 500 RPD, 15 RPM free tier
]


def _call_gemini(prompt: str, api_key: str) -> str | None:
    for model in GEMINI_MODELS:
        url = f"{GEMINI_BASE}/{model}:generateContent"
        for attempt in range(4):  # up to 4 retries per model on 429
            try:
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=30,
                )
                if resp.status_code == 404:
                    print(f"  Gemini {model} not found, trying next model...")
                    break  # next model
                if resp.status_code == 429:
                    wait = 5 * (2 ** attempt)
                    print(f"  Gemini {model} rate limited (429), retrying in {wait}s...")
                    time.sleep(wait)
                    continue  # retry same model
                resp.raise_for_status()
                return (
                    resp.json()
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
            except Exception as e:
                print(f"  Gemini {model} error ({e.__class__.__name__}): {e}, trying next model...")
                break  # next model
    return None


def match_name(target: str, candidates: list[str]) -> str | None:
    """Single-name convenience wrapper around the batch function."""
    result = match_names_batch([target], candidates)
    return result.get(target)


def match_names_batch(targets: list[str], candidates: list[str]) -> dict[str, str | None]:
    """
    One API call: match all targets against all candidates.
    Returns {target: matched_candidate_or_None}.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not targets or not candidates:
        return {t: None for t in targets}

    safe_targets = [_ascii_safe(t) for t in targets]
    safe_candidates = [_ascii_safe(c) for c in candidates]

    targets_str = "\n".join(f"- {t}" for t in safe_targets)
    candidates_str = "\n".join(f"- {c}" for c in safe_candidates)

    prompt = (
        "You are matching fighter names between two data sources. "
        "Names may differ by transliteration, East Asian name order (Family Given vs Given Family), "
        "nicknames, abbreviations, or spacing.\n\n"
        f"Tapology names (candidates):\n{candidates_str}\n\n"
        f"UFCStats names to match:\n{targets_str}\n\n"
        "Return a JSON object mapping each UFCStats name to the matching Tapology name, "
        'or null if no match exists. Example: {"Song Yadong": "Yadong Song", "John Doe": null}\n'
        "Return ONLY the JSON object, no explanation."
    )

    print(f"  [Gemini batch] {len(targets)} targets vs {len(candidates)} candidates")
    text = _call_gemini(prompt, api_key)
    if not text:
        return {t: None for t in targets}

    # Strip markdown code fences if present
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [Gemini batch] JSON parse failed: {text[:200]}")
        return {t: None for t in targets}

    # Map safe target keys back to original targets, and safe candidate values back to originals
    safe_to_orig_cand = {s: o for s, o in zip(safe_candidates, candidates)}
    result = {}
    for orig_t, safe_t in zip(targets, safe_targets):
        val = raw.get(safe_t)
        if val and isinstance(val, str):
            # val is a safe candidate string — find original
            matched = safe_to_orig_cand.get(val)
            if matched:
                print(f"  [Gemini batch] '{orig_t}' -> '{matched}'")
                result[orig_t] = matched
            else:
                result[orig_t] = None
        else:
            result[orig_t] = None

    return result
