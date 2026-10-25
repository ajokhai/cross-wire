# Policy benchmark: shipped weights on 50 unseen 2D arenas, plus CPython inference time.
#   .venv/bin/python bench/policy.py   (from the repo root)
import json, sys, time, statistics
sys.path[:0] = ["shared", "sim"]
from policy import act
from world import World
w = json.load(open("esp32/weights.json"))
q = lambda s: [round(x * 100) / 100 for x in s]
rows = []
for seed in range(1000, 1050):
    wd = World(seed=seed)
    for _ in range(250):
        wd.step(*act(w, q(wd.sensors())))
    rows.append((wd.distance, wd.collisions, len(wd.visited)))
d = [r[0] for r in rows]; c = [r[1] for r in rows]; v = [r[2] for r in rows]
print("arenas 50, 250 steps (25 s) each")
print("distance m: mean %.2f median %.2f min %.2f max %.2f" % (statistics.mean(d), statistics.median(d), min(d), max(d)))
print("collisions: mean %.2f, arenas with zero %d/50, max %d" % (statistics.mean(c), sum(1 for x in c if x == 0), max(c)))
print("cells visited: mean %.1f" % statistics.mean(v))
t = time.perf_counter(); n = 20000
for _ in range(n): act(w, [0.5, 0.6, 0.7, 0.8, 0.9])
print("cpython act(): %.1f us" % ((time.perf_counter() - t) / n * 1e6))
