import re, unicodedata
from functools import lru_cache


# --- ultra-fast normalization ---
@lru_cache(maxsize=4096)
def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(s.split())


def _initials_variant(norm_name: str) -> str:
    # norm_name is already normalized
    toks = norm_name.split()
    if not toks: return ""
    toks[0] = toks[0][:1]
    return " ".join(toks)


# Cheap token-set Sørensen–Dice (very fast in Python)
def _token_set_score(a_norm: str, b_norm: str) -> float:
    if not a_norm or not b_norm: return 0.0
    A, B = set(a_norm.split()), set(b_norm.split())
    if not A or not B: return 0.0
    inter = len(A & B)
    return (2.0 * inter) / (len(A) + len(B))


# Optional: RapidFuzz (much faster and higher quality)
try:
    from rapidfuzz import fuzz
    def _score(a_norm: str, b_norm: str) -> float:
        # token_set_ratio is robust to order and duplicates
        return fuzz.token_set_ratio(a_norm, b_norm) / 100.0
    HAVE_RF = True
except Exception:
    print(f'Optional but to increase speed, pip install rapidfuzz')
    def _score(a_norm: str, b_norm: str) -> float:
        # fallback: combine set-score + simple prefix/length heuristic
        base = _token_set_score(a_norm, b_norm)
        # light boost if prefixes match
        if a_norm and b_norm and a_norm[0] == b_norm[0]:
            base += 0.05
        return min(base, 1.0)
    HAVE_RF = False

def compare_names(name_a: str, name_b: str) -> float:
    a_norm = _norm(name_a)
    b_norm = _norm(name_b)
    if not a_norm or not b_norm:
        return 0.0

    # Token sets
    A, B = set(a_norm.split()), set(b_norm.split())
    inter = len(A & B)
    score = (2.0 * inter) / (len(A) + len(B))

    # Extra: boost if one is contained inside the other
    if a_norm in b_norm or b_norm in a_norm:
        score = max(score, 0.9)

    # If RapidFuzz available, combine with token_set_ratio
    if HAVE_RF:
        rf = fuzz.token_set_ratio(a_norm, b_norm) / 100.0
        score = max(score, rf)

    return score


class EventNameIndex:
    """Build once per event; reuse for many lookups."""
    __slots__ = ("fighters", "cands_norm", "cands_init")

    def __init__(self, event_json):
        # collect candidate fighter dicts once
        seen = set()
        fighters = []
        fdate = event_json.get('date')
        event_id = event_json.get('event_id').rstrip("/").split("/")[-1]
        for bout in event_json.get("fights", []):
            ov = bout.get("fight_overview", {}) or {}
            # pull the 3 overview fields you want
            ov_fields = {
                "fight_card_type": ov.get("fight_card_type"),
                "weight_class_weight": ov.get("weight_class_weight"),
                "weight_class": ov.get("weight_class"),
                "date": fdate,
                "event_id": event_id
            }

            for k in ("fighter1", "fighter2"):
                f = bout.get(k, {}) or {}
                n = f.get("fighter_name")
                if not n:
                    continue
                key = _norm(n)
                if key and key not in seen:
                    seen.add(key)
                    # make a copy and enrich with overview fields
                    f_enriched = dict(f)
                    f_enriched.update(ov_fields)
                    fighters.append(f_enriched)

        self.fighters = fighters
        self.cands_norm = [_norm(f["fighter_name"]) for f in fighters]
        self.cands_init = [_initials_variant(n) for n in self.cands_norm]

    def find(self, target_name: str, threshold: float = 0.82):
        """
        Returns (matched_name, score, fighter_dict)
        fighter_dict contains keys like fighter_name, img_link, nickname, etc.
        """
        if not self.fighters:
            return None, 0.0, None

        t_norm = _norm(target_name)
        if not t_norm:
            return None, 0.0, None

        # --- ultra-cheap early exits ---
        for f, nrm in zip(self.fighters, self.cands_norm):
            if nrm == t_norm:
                return f["fighter_name"], 1.0, f
        for f, nrm in zip(self.fighters, self.cands_norm):
            if nrm.startswith(t_norm) or t_norm.startswith(nrm):
                return f["fighter_name"], 0.92, f

        # initials variant once
        t_init = _initials_variant(t_norm)

        best_f, best_score = None, 0.0
        for f, nrm, init in zip(self.fighters, self.cands_norm, self.cands_init):
            s = _score(t_norm, nrm)
            s = max(s, _score(t_init, nrm), _score(t_norm, init))
            if s > best_score:
                best_f, best_score = f, s
                if best_score >= 0.97:
                    break

        if best_f and best_score >= threshold:
            return best_f["fighter_name"], best_score, best_f
        return None, best_score, None