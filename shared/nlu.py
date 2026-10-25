# On-device instruction understanding: turns plain English into calls from a robot's
# skill menu. Runs in MicroPython (ESP32) and CPython (testing).
#
# Word vectors are distilled from a sentence-embedding model (see tools/build_nlu.py)
# and stored as int8 in nlu_vec.bin, looked up by FNV-1a hash (nlu_hash.bin, sorted).
# nlu_rank.bin holds one byte per word: how rare it is (0 = very common, 255 = rare),
# so "go" counts for less than "explore" when matching.
# Skills are matched by meaning, so any robot's menu works without retraining:
#   menu line:  "go(m) drive forward or backward a distance"
#   "back up two metres then spin around" -> [("go", -2.0), ("turn", 180.0)]
import math
import struct

try:
    from array import array
except ImportError:  # pragma: no cover
    array = None

STOP = set("the a an to of please can you could would will robot then and now just for me it is be at on in by "
           "with your some bit are i am do does over".split())
UNITS_M = {"m": 1, "meter": 1, "meters": 1, "metre": 1, "metres": 1, "cm": 0.01, "centimeters": 0.01,
           "centimetres": 0.01, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048, "mm": 0.001}
NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
           "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
           "half": 0.5, "couple": 2, "few": 3, "once": 1, "twice": 2}
NEGATE = {"back", "backward", "backwards", "reverse", "behind", "right", "clockwise", "down", "close", "lower"}
SPLIT = (" and then ", " then ", ", then ", " after that ", "; ", ", and ", " and ")


def fnv1a(word):
    h = 0x811C9DC5
    for b in word.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def tokens(text):
    out, cur = [], ""
    for ch in text.lower():
        if ch.isalpha() or ch.isdigit() or ch in ".-":
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return [t.strip(".") for t in out if t.strip(".")]


def number(tok):
    if tok in NUMBERS:
        return NUMBERS[tok]
    try:
        return float(tok)
    except ValueError:
        return None


class Embeddings:
    def __init__(self, hash_path, vec_path, rank_path, dim):
        self.dim = dim
        with open(hash_path, "rb") as f:
            data = f.read()
        if array is not None:
            self.hashes = array("I", data)
        else:  # pragma: no cover
            self.hashes = list(struct.unpack("<%dI" % (len(data) // 4), data))
        with open(rank_path, "rb") as f:
            self.ranks = f.read()
        self.vec_file = open(vec_path, "rb")
        self.cache = {}

    def _index(self, h):
        lo, hi = 0, len(self.hashes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            v = self.hashes[mid]
            if v == h:
                return mid
            if v < h:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def word(self, w):
        """(unit vector, weight 0.2..1) for a word, or None if it isn't in the vocabulary."""
        if w in self.cache:
            return self.cache[w]
        found = None
        for cand in (w, w[:-1] if w.endswith("s") else None, w[:-3] if w.endswith("ing") else None,
                     w[:-2] if w.endswith("ed") else None):
            if not cand:
                continue
            i = self._index(fnv1a(cand))
            if i >= 0:
                self.vec_file.seek(i * self.dim)
                raw = self.vec_file.read(self.dim)
                vec = array("f", [(b - 256 if b > 127 else b) / 127 for b in raw])
                found = (vec, 0.2 + 0.8 * self.ranks[i] / 255)
                break
        if len(self.cache) < 40:  # each vector is 256 bytes; RAM is tight on the ESP32
            self.cache[w] = found
        return found

    def words(self, words):
        """Known content words as (vector, weight) pairs."""
        out = []
        for w in words:
            if w in STOP or w in UNITS_M or number(w) is not None:
                continue
            found = self.word(w)
            if found:
                out.append(found)
        return out

    def phrase(self, words):
        acc = [0.0] * self.dim
        found = self.words(words)
        if not found:
            return None
        for v, weight in found:
            for i in range(self.dim):
                acc[i] += v[i] * weight
        norm = math.sqrt(sum(x * x for x in acc)) or 1.0
        return array("f", [x / norm for x in acc])


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class Skill:
    def __init__(self, line, emb):
        # "go(m) drive forward or backward a distance"  /  "face(mood:happy|alert|scared|sleepy) show an emotion"
        head, _, desc = line.partition(" ")
        self.name, _, params = head.partition("(")
        params = params.rstrip(")")
        self.params = []
        for p in params.split(",") if params else ():
            kind, _, opts = p.partition(":")
            self.params.append((kind, opts.split("|") if opts else None))
        self.line = line
        desc_words = tokens(self.name + " " + desc)
        for _, opts in self.params:
            desc_words += opts or []
        self.vec = emb.phrase(desc_words)
        self.word_vecs = [v for v, _ in emb.words(desc_words)]
        self.n_numbers = sum(1 for kind, opts in self.params if not opts and kind not in ("text",))
        self.option_vecs = {}
        for kind, opts in self.params:
            for o in opts or ():
                self.option_vecs[o] = emb.phrase([o])


class Understander:
    def __init__(self, emb, menu_lines, threshold=0.5):
        self.emb = emb
        self.threshold = threshold
        self.skills = [Skill(l, emb) for l in menu_lines]

    def _args(self, skill, words, raw):
        nums = []
        for i, w in enumerate(words):
            n = number(w)
            if n is None:
                # "2m", "30cm"
                for unit, scale in UNITS_M.items():
                    if w.endswith(unit) and number(w[: -len(unit)]) is not None:
                        n = number(w[: -len(unit)]) * scale
                        break
            elif i + 1 < len(words) and words[i + 1] in UNITS_M:
                n = n * UNITS_M[words[i + 1]]
            if n is not None:
                nums.append(n)
        negative = any(w in NEGATE for w in words)
        out = []
        for kind, opts in skill.params:
            if opts:  # pick the option closest in meaning
                best, best_s = opts[0], -2
                for o in opts:
                    if o in words:
                        best = o
                        break
                    v = self.emb.phrase(words)
                    ov = skill.option_vecs.get(o)
                    s = cosine(v, ov) if v and ov else -1
                    if s > best_s:
                        best, best_s = o, s
                out.append(best)
            elif kind == "text":
                low = raw.lower()
                for verb in ("say ", "display ", "show ", "write ", "print "):
                    if verb in low:
                        out.append(raw[low.index(verb) + len(verb):].strip(" \"'"))
                        break
                else:
                    out.append(raw)
            elif kind == "deg":
                v = nums.pop(0) if nums else (180 if "around" in words else 90)
                out.append(-v if negative else v)
            elif kind in ("m", "s"):
                v = nums.pop(0) if nums else (1 if kind == "m" else 20)
                if kind == "s" and ("minute" in words or "minutes" in words):
                    v *= 60
                out.append(-v if negative and kind == "m" else v)
            else:  # generic number (x, y, ...)
                out.append(nums.pop(0) if nums else 0)
        return out

    def score(self, sk, vec, found, n_numbers):
        # 30%: whole-phrase meaning; 70%: each word's best match in the skill's description,
        # with rare words counting more than common ones
        s = cosine(vec, sk.vec)
        total = sum(w for _, w in found)
        best_words = sum(w * max(cosine(v, u) for u in sk.word_vecs) for v, w in found) / total
        s = 0.3 * s + 0.7 * best_words
        if n_numbers >= 2 and sk.n_numbers == n_numbers:  # "3, 4" fits goto(x,y)
            s += 0.15
        return s

    def clause(self, text):
        words = tokens(text)
        vec = self.emb.phrase(words)
        if vec is None:
            return None, 0.0
        found = self.emb.words(words)
        n_numbers = sum(1 for w in words if number(w) is not None)
        best, best_s = None, -1.0
        for sk in self.skills:
            if sk.vec is None or not sk.word_vecs:
                continue
            s = self.score(sk, vec, found, n_numbers)
            if s > best_s:
                best, best_s = sk, s
        return (best.name, self._args(best, words, text)), best_s

    def parse(self, text):
        """Returns (calls, confidence). calls is None when the robot should ask for help."""
        parts = [text]
        for sep in SPLIT:
            parts = [q for p in parts for q in p.split(sep)]
        calls, worst = [], 1.0
        for p in parts:
            if not p.strip():
                continue
            call, score = self.clause(p)
            worst = min(worst, score)
            if call is None or score < self.threshold:
                return None, score
            calls.append(call)
        return (calls or None), worst
