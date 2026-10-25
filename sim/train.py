# Evolve the policy weights in the headless sim, then write esp32/weights.json.
# Simple (mu, lambda) evolution strategy; no numpy needed.
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
sys.path.insert(0, HERE)

from policy import N_WEIGHTS, act  # noqa: E402
from world import World  # noqa: E402

STEPS = 250
EPISODES = 3
POP = 40
PARENTS = 8
GENERATIONS = 40


def quantize(w):
    # The robot only sees 0..100 integers over the wire, so train on the same values.
    return [round(x * 100) / 100 for x in w]


def fitness(w, seeds):
    total = 0.0
    for seed in seeds:
        world = World(seed=seed)
        for _ in range(STEPS):
            left, right = act(w, quantize(world.sensors()))
            world.step(left, right)
        # reward exploring new ground, punish bumping
        total += world.distance + 0.5 * len(world.visited) - 2.0 * world.collisions
    return total / len(seeds)


def main():
    rng = random.Random(1)
    sigma = 0.5
    parents = [[rng.gauss(0, 0.5) for _ in range(N_WEIGHTS)] for _ in range(PARENTS)]
    best_w, best_f = parents[0], float("-inf")
    for gen in range(GENERATIONS):
        seeds = [rng.randrange(10**6) for _ in range(EPISODES)]
        pop = parents + [[x + rng.gauss(0, sigma) for x in rng.choice(parents)] for _ in range(POP - PARENTS)]
        scored = sorted(((fitness(w, seeds), w) for w in pop), key=lambda t: -t[0])
        parents = [w for _, w in scored[:PARENTS]]
        if scored[0][0] > best_f:
            best_f, best_w = scored[0]
        sigma = max(0.05, sigma * 0.95)
        print("gen %2d  best %.1f  mean-top %.1f" % (gen, scored[0][0], sum(f for f, _ in scored[:PARENTS]) / PARENTS), flush=True)

    # score the winner on fresh arenas it never trained on
    test = [World(seed=s) for s in (11, 22, 33, 44, 55)]
    for world in test:
        for _ in range(STEPS):
            world.step(*act(best_w, quantize(world.sensors())))
    print("held-out: distance %.1f m, collisions %d, cells %d (avg over 5)" % (
        sum(w.distance for w in test) / 5, sum(w.collisions for w in test) / 5, sum(len(w.visited) for w in test) / 5))

    out = os.path.join(HERE, "..", "esp32", "weights.json")
    with open(out, "w") as f:
        json.dump([round(x, 4) for x in best_w], f)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
