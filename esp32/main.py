# Robot brain for a wheeled robot (the Husky in the sim). Everything the robot decides
# happens here; the sim only supplies sensors and applies wheel commands.
#
# Over USB serial, as Cross Talk frames (relayed by the bridge):
#   sim -> brain   DM "s <seq> <d1..d5> <x_cm> <y_cm> <heading_deg>"   ranges 0..100 (100 = clear),
#                                                                     at -60 -30 0 30 60 degrees
#   brain -> sim   DM "m <seq> <left> <right>"                         wheel speeds -100..100
#   anyone         "@husky <instruction>" in the channel, or a DM; "!repl" gives the REPL back
#
# Instructions: exact skill calls ("go 2; turn -90") run as they are. Plain English goes
# through the on-chip command model (nlu.py). If it isn't sure, the robot posts its skills
# and asks "@pilot" (a language model elsewhere on the robot), whose reply comes back as
# exact calls. The BOOT button stops the brain.
import gc
import json
import math
import select
import sys
import time

import micropython
from machine import I2C, Pin

import ssd1306
from crosstalk_micro import CrossTalkMicro as XT, FrameReader, OP_BROADCAST, OP_DIRECT_MSG
from face import Face
from nlu import Embeddings, Understander
from policy import act

NAME = "husky"
PILOT = "pilot"
SIM = "robot-sim"
MENU = [
    "go(m) drive forward, or back up in reverse, a distance in meters",
    "turn(deg) turn, spin or rotate in place by an angle, left is positive",
    "goto(x,y) go to, navigate or head to a position x y on the map",
    "explore(s) wander, roam and explore by itself while avoiding obstacles",
    "stop() stop moving, halt, freeze",
    "face(mood:happy|alert|scared|sleepy) show an emotion on its face",
    "say(text) show a message on its screen",
    "status() report where it is, its position and what it is doing",
]
MENU_POST = [
    NAME + " skills: go(m) turn(deg) goto(x,y) explore(s) stop() face(mood) say(text) status()",
    NAME + " args: m metres (- = back), deg (+ = left), x y metres, s seconds, mood happy|alert|scared|sleepy",
]
MAX_GO = 10.0      # metres; refuse anything bigger, whoever asks
MAX_GOTO = 20.0    # metres from where the robot is
IDLE_MS = 2000     # no sensor data for this long -> sleepy face
STATUS_MS = 3000   # heartbeat to the channel
FACE_MS = 40       # face redraw interval
HOLD_MS = 15000    # how long a face/text set over chat stays


def hypot(dx, dy):  # MicroPython's math has no hypot
    return math.sqrt(dx * dx + dy * dy)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def fmt(call):
    name, args = call
    return name + "".join(" %g" % a if isinstance(a, (int, float)) else " " + str(a) for a in args)


def exact(text):
    """'go 2; turn -90' or 'go(2)' -> calls, or None if any part isn't a menu call."""
    calls = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name = part.split("(")[0].split()[0].lower() if part.split() else ""
        sk = None
        for s in nlu.skills:
            if s.name == name:
                sk = s
        if sk is None:
            return None
        rest = part[len(name):].strip()
        if rest.startswith("(") and rest.endswith(")"):
            rest = rest[1:-1]
        args = []
        if sk.params and sk.params[0][0] == "text":
            args = [rest.strip(" \"'")]
        else:
            vals = rest.replace(",", " ").split()
            if len(vals) > len(sk.params):
                return None
            for (kind, opts), v in zip(sk.params, vals):
                if opts:
                    if v not in opts:
                        return None
                    args.append(v)
                else:
                    try:
                        args.append(float(v))
                    except ValueError:
                        return None
            if len(args) < len(sk.params):
                if name == "goto":
                    return None  # needs both coordinates
                args += [None] * (len(sk.params) - len(args))  # filled with defaults later
        calls.append((name, args))
    return calls or None


class Skills:
    """Runs skill calls one after another, closed-loop on odometry, with the policy net as reflexes."""

    def __init__(self, post):
        self.post = post
        self.queue = []
        self.cur = None
        self.status = "idle"

    def run(self, calls):
        self.queue.extend(calls)

    def stop(self, why="stopped"):
        self.queue = []
        self.cur = None
        self.status = why

    def _begin(self, call, pose, now):
        name, args = call
        if name == "stop":
            self.stop()
            self.post("stopped")
            return None
        if name == "face":
            face.set(mood=args[0])
            self.post("face " + args[0])
            return None
        if name == "say":
            face.set(caption=args[0])
            self.post("showing: " + args[0][:40])
            return None
        if name == "status":
            x, y, h = pose
            self.post("at %.1f, %.1f facing %d deg; %s; %d queued" % (x, y, math.degrees(h), self.status, len(self.queue)))
            return None
        defaults = {"go": 1.0, "turn": 90.0, "explore": 20.0, "face": "happy"}
        if name in defaults and (not args or args[0] is None):
            args = [defaults[name]]
        x, y, _ = pose
        if (name == "go" and abs(args[0]) > MAX_GO) or (name == "goto" and hypot(args[0] - x, args[1] - y) > MAX_GOTO):
            self.post("%s fail too far" % fmt((name, args)))
            return None
        self.status = fmt((name, args))
        face.set(caption=self.status)
        return {"name": name, "args": args, "pose": pose, "t": now}

    def _finish(self, ok, why=""):
        self.post(self.cur["name"] + (" done" if ok else " fail " + why))
        self.cur = None
        self.status = "idle" if not self.queue else self.status

    def step(self, ranges, pose, now):
        """One control tick. Returns wheel speeds (left, right) in -1..1."""
        while self.cur is None and self.queue:
            self.cur = self._begin(self.queue.pop(0), pose, now)
        c = self.cur
        if c is None:
            return 0.0, 0.0
        x, y, h = pose
        x0, y0, h0 = c["pose"]
        elapsed = time.ticks_diff(now, c["t"]) / 1000
        front = min(ranges[1:4])
        name, args = c["name"], c["args"]

        if name == "go":
            d = args[0]
            sign = 1 if d >= 0 else -1
            remaining = abs(d) - hypot(x - x0, y - y0)
            if remaining <= 0.03:
                self._finish(True)
                return 0.0, 0.0
            if sign > 0 and front < 0.2:
                self._finish(False, "blocked")
                return 0.0, 0.0
            if elapsed > abs(d) * 3 + 5:
                self._finish(False, "stuck")
                return 0.0, 0.0
            speed = sign * clamp(remaining / 0.6 + 0.15, 0.15, 0.7)
            turn = clamp(2.0 * wrap(h0 - h), -0.4, 0.4)
            return speed - turn, speed + turn

        if name == "turn":
            err = wrap(h0 + math.radians(args[0]) - h) if abs(args[0]) < 180 else None
            if err is None:  # 180+ degrees: track how far we've turned instead of the target heading
                c.setdefault("turned", 0.0)
                c["turned"] += wrap(h - c.get("last", h0))
                c["last"] = h
                err = math.radians(args[0]) - c["turned"]
            if abs(err) < math.radians(3):
                self._finish(True)
                return 0.0, 0.0
            if elapsed > 20:
                self._finish(False, "stuck")
                return 0.0, 0.0
            w = clamp(1.5 * err, -0.8, 0.8)
            if abs(w) < 0.3:
                w = 0.3 if w > 0 else -0.3
            return -w, w

        if name == "goto":
            dx, dy = args[0] - x, args[1] - y
            if hypot(dx, dy) < 0.3:
                self._finish(True)
                return 0.0, 0.0
            if elapsed > 120:
                self._finish(False, "timeout")
                return 0.0, 0.0
            if min(ranges) < 0.45:  # something close: let the reflex net steer around it
                return act(weights, ranges)
            err = wrap(math.atan2(dy, dx) - h)
            fwd = 0.7 * clamp(1 - abs(err) / 1.2, 0, 1)
            turn = clamp(1.5 * err, -0.6, 0.6)
            return fwd - turn, fwd + turn

        if name == "explore":
            if elapsed > args[0]:
                self._finish(True)
                return 0.0, 0.0
            return act(weights, ranges)

        self._finish(False, "unknown skill")
        return 0.0, 0.0


def mood_for(ranges, busy):
    nearest = min(ranges)
    if nearest < 0.15:
        return "scared"
    if nearest < 0.4:
        return "alert"
    return "happy"


boot_btn = Pin(0, Pin.IN, Pin.PULL_UP)
oled = ssd1306.SSD1306_I2C(128, 64, I2C(0, sda=Pin(21), scl=Pin(22), freq=400000))
face = Face(oled)
face.set(caption="loading words")
face.draw()
with open("weights.json") as f:
    weights = json.load(f)
nlu = Understander(Embeddings("nlu_hash.bin", "nlu_vec.bin", "nlu_rank.bin", 64), MENU)
gc.collect()


def run():
    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)
    reader = FrameReader(max_frame=256)  # pilot plans can be longer than the default 128

    def post(text):
        data = text.encode()[:110]
        out.write(XT.pack_text(data.decode("utf-8", "ignore")))

    skills = Skills(post)
    micropython.kbd_intr(-1)  # 0x03 bytes are data now, not Ctrl-C
    out.write(XT.pack_register(NAME))

    now = time.ticks_ms()
    last_data = time.ticks_add(now, -IDLE_MS)
    last_status = now
    last_face = now
    held_until = now

    menu_posted = [time.ticks_add(time.ticks_ms(), -60000)]

    def instruct(sender, text, is_dm):
        calls = exact(text)
        how = "exact"
        if calls is None and sender == PILOT:
            # the pilot must answer with exact calls; never guess at machine-written plans
            post("can't run pilot plan: " + text[:60])
            return
        if calls is None:
            calls, conf = nlu.parse(text)
            how = "understood %.2f" % conf
        if calls is None:
            if time.ticks_diff(time.ticks_ms(), menu_posted[0]) > 60000:
                for line in MENU_POST:
                    post(line)
                menu_posted[0] = time.ticks_ms()
            post("not sure what that means; asking %s" % PILOT)
            out.write(XT.pack_dm(PILOT, "%s asks %s: %s" % (sender, NAME, text[:70]), False))
            face.set(mood="alert", caption="asking pilot")
            return
        if any(n not in ("face", "say", "status") for n, _ in calls) and calls[0][0] != "stop":
            skills.stop("replanned")  # a new movement order replaces the old plan
        if calls[0][0] == "stop":
            skills.stop()
        skills.run(calls)
        post("ok (%s): %s" % (how, "; ".join(fmt(c) for c in calls)))

    while boot_btn.value():
        buf = bytearray()
        while len(buf) < 512 and poll.poll(0):
            buf += inp.read(1)

        for frame in reader.feed(buf) if buf else ():
            if frame["opcode"] not in (OP_BROADCAST, OP_DIRECT_MSG):
                continue
            parsed = XT.unpack_attributed(frame["payload"])
            if not parsed:
                continue
            sender, text = parsed
            text = text.strip()
            if text == "!repl":
                return
            parts = text.split()
            if sender == SIM:  # sensor data only; a mangled one is dropped, never read as an instruction
                if len(parts) != 10 or parts[0] != "s":
                    continue
                try:
                    ranges = [int(p) / 100 for p in parts[2:7]]
                    pose = (int(parts[7]) / 100, int(parts[8]) / 100, math.radians(int(parts[9])))
                except ValueError:
                    continue
                now = time.ticks_ms()
                try:
                    left, right = skills.step(ranges, pose, now)
                except Exception as err:  # report it and stop safely instead of dying
                    skills.stop("error")
                    post("error in skill: %r" % err)
                    left = right = 0.0
                out.write(XT.pack_dm(SIM, "m %s %d %d" % (parts[1], int(left * 100), int(right * 100)), False))
                if time.ticks_diff(now, held_until) > 0:
                    face.set(mood=mood_for(ranges, skills.cur), look=(left - right) * 1.5,
                             caption=skills.status if skills.cur else "ready")
                last_data = now
                continue

            is_dm = frame["opcode"] == OP_DIRECT_MSG
            low = text.lower()
            if is_dm or sender == PILOT:
                # models like to address the robot by name: "@husky ...", "husky: ..."
                for prefix in ("@" + NAME, NAME + ":", NAME):
                    if low.startswith(prefix):
                        text = text[len(prefix):].strip(" :,")
                        break
                if not is_dm and text == parsed[1].strip():
                    continue  # pilot talking to someone else
            elif low.startswith("@" + NAME):
                text = text[len(NAME) + 1:].strip(" :,")
            else:
                continue  # channel chatter not meant for us
            if text in ("skills", "help", "?"):
                for line in MENU_POST:
                    post(line)
                continue
            try:
                instruct(sender, text, is_dm)
            except Exception as err:  # a bad message must never kill the brain
                post("error: %r" % err)
            held_until = time.ticks_add(time.ticks_ms(), 3000)

        now = time.ticks_ms()
        if time.ticks_diff(now, last_data) > IDLE_MS and time.ticks_diff(now, held_until) > 0 and face.mood != "sleepy":
            face.set(mood="sleepy", look=0, caption="waiting for sim")
        if time.ticks_diff(now, last_status) > STATUS_MS:
            out.write(XT.pack_heartbeat(skills.status))
            last_status = now
        if time.ticks_diff(now, last_face) > FACE_MS:
            face.draw()
            last_face = now


if boot_btn.value():
    try:
        run()
    finally:
        micropython.kbd_intr(3)
        face.set(mood="boot", look=0, caption="REPL mode")
        face.draw()
else:
    face.set(mood="boot", caption="REPL mode")
    face.draw()
