# Distil word vectors for the on-device command model (shared/nlu.py) and score it on
# skill menus for three different robots. Writes esp32/nlu_hash.bin, nlu_vec.bin, nlu_rank.bin.
#
#   .venv/bin/python tools/build_nlu.py [--model minishlab/potion-base-8M] [--dim 64] [--words 6000] [--eval-only]
import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "shared"))
from husky_menu import MENU as HUSKY, OBJECT_WORDS  # noqa: E402
from nlu import Embeddings, Understander, fnv1a  # noqa: E402

MENUS = {
    "husky": HUSKY,
    "arm": [
        "reach(x,y,z) move the hand to a point",
        "grip() close the gripper claw to grab or pick up an object",
        "release() open the gripper to drop or let go of an object",
        "home() return the arm to its rest position",
        "wave() wave hello",
        "stop() stop moving, halt, freeze",
    ],
    "usv": [
        "thrust(m) sail or motor forward a distance in meters",
        "heading(deg) steer the boat to a new heading",
        "anchor() drop anchor and hold position",
        "patrol(s) patrol the area by itself",
        "stop() cut the engines and stop",
    ],
}

TESTS = [
    ("husky", "drive forward two meters", "go"), ("husky", "move ahead 3 m", "go"),
    ("husky", "back up a bit", "go"), ("husky", "reverse 50 cm", "go"),
    ("husky", "spin around", "turn"), ("husky", "rotate 45 degrees to the right", "turn"),
    ("husky", "turn left", "turn"), ("husky", "turn around", "turn"), ("husky", "head over to 3, 4", "goto"),
    ("husky", "navigate to x 2 y -1", "goto"), ("husky", "go explore for a minute", "explore"),
    ("husky", "roam around the room", "explore"), ("husky", "wander", "explore"),
    ("husky", "halt!", "stop"), ("husky", "freeze", "stop"), ("husky", "stop right there", "stop"),
    ("husky", "look happy", "face"), ("husky", "act scared", "face"), ("husky", "be sleepy", "face"),
    ("husky", "say hello world", "say"), ("husky", "display hi josh", "say"),
    ("husky", "where are you", "status"), ("husky", "report your position", "status"),
    ("husky", "investigate that blue square", "approach"), ("husky", "check out the red pillar", "approach"),
    ("husky", "drive up to the orange barrel", "approach"), ("husky", "inspect the green box", "approach"),
    ("husky", "find the green cube", "find"), ("husky", "search for something yellow", "find"),
    ("husky", "look for the blue box", "find"),
    ("husky", "go back to where you started", "return"), ("husky", "return to the start", "return"),
    ("husky", "what can you see", "look"), ("husky", "describe what's around you", "look"),
    ("arm", "grab the cube", "grip"), ("arm", "pick it up", "grip"), ("arm", "close the claw", "grip"),
    ("arm", "let go", "release"), ("arm", "drop it", "release"), ("arm", "open the hand", "release"),
    ("arm", "go back to rest", "home"), ("arm", "wave hello", "wave"),
    ("arm", "move the hand to 0.3 0.1 0.2", "reach"), ("arm", "stop", "stop"),
    ("usv", "sail forward 100 meters", "thrust"), ("usv", "steer to heading 90", "heading"),
    ("usv", "drop anchor", "anchor"), ("usv", "hold position here", "anchor"),
    ("usv", "patrol the harbor", "patrol"), ("usv", "kill the engines", "stop"),
]
# None = don't check that argument (object descriptions are resolved by the robot)
MULTI = [
    ("husky", "drive forward two meters then turn right", [("go", [2.0]), ("turn", [-90])]),
    ("husky", "back up 1 m and spin around", [("go", [-1.0]), ("turn", [180])]),
    ("husky", "go to 3, 4 then look happy", [("goto", [3.0, 4.0]), ("face", ["happy"])]),
    ("husky", "turn around, investigate that blue square, then go back to where you started",
     [("turn", [180]), ("approach", [None]), ("return", [])]),
]


def build(model_name, dim, n_words, out_dir):
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(model_name)
    extra = set()
    for menu in MENUS.values():
        for line in menu:
            extra.update(w.strip("(),:|").lower() for w in line.replace("(", " ").replace("|", " ").split())
    # the tokenizer's whole-word entries run roughly from most to least common
    vocab = model.tokenizer.get_vocab()
    words = [t for t in sorted(vocab, key=vocab.get) if t.isalpha() and t.isascii() and len(t) > 1][:n_words]
    rank = {w: i for i, w in enumerate(words)}
    words = sorted(set(words) | {w for w in extra if w.isalpha()})
    vecs = model.encode(words)
    vecs = vecs - vecs.mean(0)
    _, _, vt = np.linalg.svd(vecs, full_matrices=False)  # PCA
    low = vecs @ vt[:dim].T
    low /= np.linalg.norm(low, axis=1, keepdims=True) + 1e-9
    q = np.clip(np.round(low * 127), -127, 127).astype(np.int8)
    order = sorted(range(len(words)), key=lambda i: fnv1a(words[i]))
    hashes = [fnv1a(words[i]) for i in order]
    assert len(set(hashes)) == len(hashes), "hash collision"
    with open(os.path.join(out_dir, "nlu_hash.bin"), "wb") as f:
        f.write(struct.pack("<%dI" % len(hashes), *hashes))
    with open(os.path.join(out_dir, "nlu_vec.bin"), "wb") as f:
        f.write(q[order].tobytes())
    with open(os.path.join(out_dir, "nlu_rank.bin"), "wb") as f:
        f.write(bytes(int(255 * np.log1p(rank.get(words[i], n_words)) / np.log1p(n_words)) for i in order))
    return len(words)


def same(got, want):
    if len(got) != len(want):
        return False
    for (gn, ga), (wn, wa) in zip(got, want):
        if gn != wn or len(ga) != len(wa):
            return False
        for g, w in zip(ga, wa):
            if w is not None and (round(g, 2) if isinstance(g, float) else g) != w:
                return False
    return True


def evaluate(out_dir, dim, verbose):
    emb = Embeddings(os.path.join(out_dir, "nlu_hash.bin"), os.path.join(out_dir, "nlu_vec.bin"),
                     os.path.join(out_dir, "nlu_rank.bin"), dim)
    bots = {name: Understander(emb, menu, object_words=OBJECT_WORDS if name == "husky" else ()) for name, menu in MENUS.items()}
    right = 0
    for bot, text, want in TESTS:
        call, score = bots[bot].clause(text)
        ok = call and call[0] == want
        right += bool(ok)
        if verbose or not ok:
            print("  %s %-6s %-40s -> %-30s %.2f" % ("ok " if ok else "BAD", bot, text, call, score))
    multi_ok = 0
    for bot, text, want in MULTI:
        calls, conf = bots[bot].parse(text)
        ok = same(calls or [], want)
        multi_ok += ok
        print("  %s multi %s\n        -> %s (%.2f)" % ("ok " if ok else "BAD", text, calls, conf))
    print("single: %d/%d   multi-step: %d/%d" % (right, len(TESTS), multi_ok, len(MULTI)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minishlab/potion-base-8M")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--words", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(ROOT, "esp32"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not args.eval_only:
        n = build(args.model, args.dim, args.words, args.out)
        size = sum(os.path.getsize(os.path.join(args.out, f)) for f in ("nlu_vec.bin", "nlu_hash.bin", "nlu_rank.bin"))
        print("%s dim=%d: %d words, %d KB on flash" % (args.model, args.dim, n, size // 1024))
    evaluate(args.out, args.dim, args.verbose)


if __name__ == "__main__":
    main()
