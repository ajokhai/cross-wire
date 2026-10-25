# Runs the MuJoCo world for a URDF robot and lets the ESP32 drive it. It also puts the
# ESP32 into a Cross Talk channel (both share the one USB port), so people and agents
# can instruct it with "@husky ...".
#
#   .venv/bin/mjpython sim/mj_sim.py --channel xt_...            # with the 3D viewer
#   .venv/bin/python   sim/mj_sim.py --channel xt_... --headless
#   .venv/bin/python   sim/mj_sim.py --headless --say "go 2; turn 90" --seconds 20   # no hub
import argparse
import math
import os
import queue
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bridge"))

from mj_world import CONTROL_DT, RobotWorld  # noqa: E402
from xt_bridge import DeviceBridge  # noqa: E402

SIM = "robot-sim"
REPLY_TIMEOUT = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.path.join(ROOT, "robots", "husky", "husky.urdf"))
    ap.add_argument("--port", default="/dev/cu.usbserial-210")
    ap.add_argument("--channel", help="Cross Talk channel for the robot (omit to run without a hub)")
    ap.add_argument("--name", default="husky", help="the robot's agent name (must match the brain)")
    ap.add_argument("--url", default=os.environ.get("CROSSTALK_URL", "ws://localhost:4488"))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--say", action="append", default=[], help="instruction to give the robot at start (repeatable)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this much sim time (0 = forever)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    world = RobotWorld(args.urdf, seed=args.seed)
    bridge = DeviceBridge(args.port, args.channel, args.name, args.url, local_names={SIM})
    bridge.send_local("bridge", "@%s skills" % args.name, dm=False)
    for text in args.say:
        bridge.send_local("tester", "@%s %s" % (args.name, text), dm=False)
    print("robot: %s, wheels %s" % (os.path.basename(args.urdf), [n for n, _ in world.wheels]), flush=True)

    # Flow control: one sensor message in flight at a time. While the brain is busy (it
    # pauses to understand an instruction), new readings would overflow its ~256-byte
    # serial buffer, so the sim waits for the answer, or 2 s, before sending another.
    stats = {"answered": 0, "missed": 0, "seq": 0, "waiting": None, "sent": 0.0}

    def tick():
        if stats["waiting"] is None or time.monotonic() - stats["sent"] > 2.0:
            x, y, h = world.pose()
            ranges = [int(round(r * 100)) for r in world.ranges()]
            stats["seq"] += 1
            stats["waiting"], stats["sent"] = str(stats["seq"]), time.monotonic()
            bridge.send_local(SIM, "s %s %s %d %d %d" % (stats["seq"], " ".join(map(str, ranges)),
                                                       round(x * 100), round(y * 100), round(math.degrees(h))))
        command = None
        deadline = time.monotonic() + REPLY_TIMEOUT
        while command is None:
            try:
                reply = bridge.local_inbox.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            parts = reply.split()
            if len(parts) == 4 and parts[0] == "m" and parts[1] == stats["waiting"]:
                command = (int(parts[2]) / 100, int(parts[3]) / 100)
        if command is None:
            stats["missed"] += 1
            world.step(0, 0)  # no answer: stop rather than drive blind
        else:
            stats["answered"] += 1
            stats["waiting"] = None
            world.step(*command)
        if stats["seq"] % 50 == 0:
            x, y, h = world.pose()
            print("t=%5.1fs  pose %.2f, %.2f, %4.0f deg  answered %d missed %d  bumps %d  [%s]" % (
                world.time, x, y, math.degrees(h), stats["answered"], stats["missed"], world.bumps, bridge.status),
                flush=True)

    def done():
        return args.seconds and world.time >= args.seconds

    if args.headless:
        while not done():
            tick()
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = world.base_id
        viewer.cam.distance, viewer.cam.elevation = 6.0, -35
        next_tick = time.monotonic()
        while viewer.is_running() and not done():
            tick()
            viewer.sync()
            next_tick += CONTROL_DT
            time.sleep(max(0.0, next_tick - time.monotonic()))


if __name__ == "__main__":
    main()
