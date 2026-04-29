import subprocess
import json
import os
import unicodedata


def _ascii_safe(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", errors="replace").decode("ascii")


def match_names_batch(targets: list[str], candidates: list[str]) -> dict[str, str | None]:
    """
    One claude -p call: match all targets against all candidates.
    Returns {target: matched_candidate_or_None}.
    """
    if not targets or not candidates:
        return {t: None for t in targets}

    safe_targets = [_ascii_safe(t) for t in targets]
    safe_candidates = [_ascii_safe(c) for c in candidates]

    targets_str = "\n".join(f"- {t}" for t in safe_targets)
    candidates_str = "\n".join(f"- {c}" for c in safe_candidates)

    prompt = (
        "Match each name in LIST_A to the same person in LIST_B. "
        "Names may differ by transliteration, word order (Family Given vs Given Family), spacing, or abbreviation. "
        "Output ONLY a valid JSON object — no explanation, no markdown. "
        'Format: {"Name A": "Matched Name B", "Other Name A": null}\n\n'
        f"LIST_A (to match):\n{targets_str}\n\n"
        f"LIST_B (candidates):\n{candidates_str}\n\n"
        "JSON output:"
    )

    CLAUDE_CMD = "/home/caramey/.npm-global/bin/claude"
    print(f"  [Claude batch] {len(targets)} targets vs {len(candidates)} candidates")
    try:
        result = subprocess.run(
            f'"{CLAUDE_CMD}" -p --no-session-persistence',
            input=prompt,
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace",
            shell=True,
            cwd=os.path.expanduser("~"),
        )
        text = (result.stdout or "").strip()
    except subprocess.TimeoutExpired:
        print("  [Claude batch] timeout")
        return {t: None for t in targets}
    except Exception as e:
        print(f"  [Claude batch] subprocess error: {e}")
        return {t: None for t in targets}

    if not text:
        print(f"  [Claude batch] empty response (stderr: {result.stderr[:100]})")
        return {t: None for t in targets}

    # Strip markdown fences if present
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [Claude batch] JSON parse failed: {text[:200]}")
        return {t: None for t in targets}

    safe_to_orig_cand = {s: o for s, o in zip(safe_candidates, candidates)}
    result_map = {}
    for orig_t, safe_t in zip(targets, safe_targets):
        val = raw.get(safe_t)
        if val and isinstance(val, str):
            matched = safe_to_orig_cand.get(val)
            if matched:
                print(f"  [Claude batch] '{orig_t}' -> '{matched}'")
                result_map[orig_t] = matched
            else:
                result_map[orig_t] = None
        else:
            result_map[orig_t] = None

    return result_map


def match_name(target: str, candidates: list[str]) -> str | None:
    result = match_names_batch([target], candidates)
    return result.get(target)
