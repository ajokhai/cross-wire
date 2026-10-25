# Full Cross Talk path: sim agent -> hub -> bridge -> USB -> ESP32 -> back, 10 Hz, while a
# second agent gives the robot plain-English instructions in the channel.
#   crosstalk serve --no-rate-limit --port 4489
#   .venv/bin/python bench/hub_loop.py 900 result.json   (from the repo root)
import json, math, os, queue, statistics, sys, threading, time
ROOT = os.getcwd()
sys.path[:0] = [os.path.join(ROOT, "sim"), os.path.join(ROOT, "bridge")]
from crosstalk import CrossTalk
from mj_world import RobotWorld
from xt_bridge import DeviceBridge
URL = "ws://localhost:4489"
STEPS = int(sys.argv[1]); OUT = sys.argv[2]
SAYS = ["status", "drive forward two meters", "turn left", "head over to 3, 4", "wander",
        "look happy", "display hi josh", "freeze", "spin around", "where are you", "explore 30"]
sim = CrossTalk.connect("robot-sim", url=URL)
ch = sim.create_channel("cross-wire-bench", topic="ESP32 brain benchmark")
inbox = queue.Queue()
sim.on("dm", lambda m: m.get("from", {}).get("name") == "husky" and inbox.put((time.monotonic(), m.get("content", ""))))
log = []
br = DeviceBridge("/dev/cu.usbserial-210", ch.address, "husky", URL, log=lambda s: log.append((time.monotonic(), s)))
op = CrossTalk.connect("operator", url=URL); op.join_channel(ch.address)
said, replies = [], []
op.on("message", lambda m: m.get("from", {}).get("name") == "husky" and replies.append((time.monotonic(), m.get("content", ""))))
time.sleep(8)
world = RobotWorld(os.path.join(ROOT, "robots/husky/husky.urdf"), seed=7)
t0 = time.monotonic()
def operator():
    for text in SAYS:
        time.sleep(5)
        said.append((time.monotonic(), text)); op.send(ch.address, "@husky " + text)
threading.Thread(target=operator, daemon=True).start()
rtts, missed = [], 0
for seq in range(1, STEPS + 1):
    tick = time.monotonic(); x, y, h = world.pose()
    r = [int(round(v * 100)) for v in world.ranges()]
    sent = time.monotonic()
    sim.dm("husky", "s %d %s %d %d %d" % (seq, " ".join(map(str, r)), round(x*100), round(y*100), round(math.degrees(h))), reply_expected=False)
    cmd = None
    while cmd is None:
        left = 0.25 - (time.monotonic() - sent)
        if left <= 0: break
        try: t, rep = inbox.get(timeout=left)
        except queue.Empty: break
        p = rep.split()
        if len(p) == 4 and p[0] == "m" and p[1] == str(seq):
            cmd = (int(p[2]) / 100, int(p[3]) / 100); rtts.append((t - sent) * 1000)
    if cmd is None: missed += 1; world.step(0, 0)
    else: world.step(*cmd)
    time.sleep(max(0, 0.1 - (time.monotonic() - tick)))
time.sleep(1)
rtts.sort(); q = lambda f: rtts[min(len(rtts) - 1, int(f * len(rtts)))]
# first husky reply after each instruction
acks = []
for ts, text in said:
    nxt = [(t, c) for t, c in replies if t > ts]
    acks.append(dict(say=text, ms=round((nxt[0][0] - ts) * 1000) if nxt else None, reply=nxt[0][1] if nxt else None))
res = dict(steps=STEPS, answered=len(rtts), missed=missed, rtt_ms=dict(min=round(rtts[0], 1), p50=round(q(.5), 1), p90=round(q(.9), 1),
           p99=round(q(.99), 1), max=round(rtts[-1], 1), mean=round(statistics.mean(rtts), 1)) if rtts else None,
           bumps=world.bumps, acks=acks, channel=ch.address)
print(json.dumps(res, indent=1))
json.dump(dict(res, rtts=rtts, replies=[(round(t - t0, 2), c) for t, c in replies]), open(OUT, "w"))
for t, c in replies: print("%6.2f husky: %s" % (t - t0, c))
os._exit(0)
