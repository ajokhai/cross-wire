# Closed-loop benchmark: MuJoCo Husky <-> ESP32 brain over USB serial (Cross Talk frames, no hub).
#   .venv/bin/python bench/serial_loop.py --steps 600 --say "explore 55" --out result.json   (from the repo root)
# Pass --channel to also join the brain to a Cross Talk hub channel (hub_loop.py is the full hub test).
import argparse, json, math, os, queue, statistics, sys, time
ROOT = os.getcwd()
sys.path[:0] = [os.path.join(ROOT, "sim"), os.path.join(ROOT, "bridge")]
from mj_world import RobotWorld
from xt_bridge import DeviceBridge
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--channel")
ap.add_argument("--say", action="append", default=[])
ap.add_argument("--out")
a = ap.parse_args()
log = []
world = RobotWorld(os.path.join(ROOT, "robots/husky/husky.urdf"), seed=7)
br = DeviceBridge("/dev/cu.usbserial-210", a.channel, "husky", "ws://localhost:4488", local_names={"robot-sim"},
                  log=lambda s: log.append((time.monotonic(), s)))
time.sleep(8.0)
t0 = time.monotonic()
for text in a.say:
    br.send_local("tester", "@husky %s" % text, dm=False)
rtts, missed, stale = [], 0, 0
for seq in range(1, a.steps + 1):
    tick = time.monotonic()
    x, y, h = world.pose()
    r = [int(round(v * 100)) for v in world.ranges()]
    sent = time.monotonic()
    br.send_local("robot-sim", "s %d %s %d %d %d" % (seq, " ".join(map(str, r)), round(x*100), round(y*100), round(math.degrees(h))))
    cmd = None
    while cmd is None:
        left = 0.25 - (time.monotonic() - sent)
        if left <= 0: break
        try: rep = br.local_inbox.get(timeout=left)
        except queue.Empty: break
        p = rep.split()
        if len(p) == 4 and p[0] == "m" and p[1] == str(seq):
            cmd = (int(p[2]) / 100, int(p[3]) / 100); rtts.append((time.monotonic() - sent) * 1000)
        else: stale += 1
    if cmd is None: missed += 1; world.step(0, 0)
    else: world.step(*cmd)
    time.sleep(max(0, 0.1 - (time.monotonic() - tick)))
wall = time.monotonic() - t0
x, y, h = world.pose()
rtts.sort()
q = lambda f: rtts[min(len(rtts) - 1, int(f * len(rtts)))]
res = dict(steps=a.steps, answered=len(rtts), missed=missed, stale=stale, wall_s=round(wall, 1), sim_s=round(world.time, 1),
           rtt_ms=dict(min=round(rtts[0], 1), p50=round(q(.5), 1), p90=round(q(.9), 1), p99=round(q(.99), 1), max=round(rtts[-1], 1), mean=round(statistics.mean(rtts), 1)),
           pose=[round(x, 2), round(y, 2), round(math.degrees(h))], bumps=world.bumps, status=br.status)
print(json.dumps(res, indent=1))
for t, s in log: print("%7.2f  %s" % (t - t0, s))
if a.out: json.dump(dict(res, rtts=rtts, log=[(round(t - t0, 3), s) for t, s in log]), open(a.out, "w"))
