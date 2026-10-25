# Robot brain for a wheeled robot (the Husky in the sim). Everything the robot decides
# happens here; the sim only supplies sensors and applies wheel commands.
#
# Over USB serial, as Cross Talk frames (relayed by the bridge):
#   sim -> brain   DM "s <seq> <d1..d5> <x_cm> <y_cm> <heading_deg>"   ranges 0..100 (100 = clear)
#                  DM "o <id> <color> <shape> <x_cm> <y_cm> <size_cm>"  an object the camera sees now
#   brain -> sim   DM "m <seq> <left> <right>"                         wheel speeds -100..100
#   anyone         "@husky <instruction>" in the channel, or a DM; "!repl" gives the REPL back
#
# Instructions: exact skill calls ("go 2; approach blue box") run as they are. Plain English
# goes through the on-chip command model (nlu.py). If it isn't sure, the robot DMs "@pilot"
# (a language model elsewhere on the robot), whose reply must be exact calls.
# The robot remembers where it started and every object it has seen, so it can resolve
# "that blue square" and "go back to where you started" itself. The BOOT button stops it.
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
from husky_menu import COLORS, MENU, MENU_POST, NAME, OBJECT_WORDS, SHAPES
from nlu import Embeddings, Understander
from policy import act

PILOT = "pilot"
SIM = "robot-sim"
SENSOR_ANGLES = (-60, -30, 0, 30, 60)  # range sensor directions, degrees (+ = left), as the sim sends them
MAX_GO = 10.0      # metres; refuse anything bigger, whoever asks
MAX_GOTO = 20.0    # metres from where the robot is
SEEN_MS = 1500     # an object counts as "in view" if the camera reported it this recently
IDLE_MS = 2000     # no sensor data for this long -> sleepy face
STATUS_MS = 3000   # heartbeat to the channel
FACE_MS = 40       # face redraw interval


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


def direction(bearing_deg):
    b = abs(bearing_deg)
    side = "left" if bearing_deg > 0 else "right"
    if b < 15:
        return "ahead"
    if b < 50:
        return "ahead-" + side
    if b < 120:
        return side
    return "behind-" + side


class Memory:
    """Where the robot started and what it has seen."""

    def __init__(self):
        self.home = None          # (x, y, heading)
        self.objects = {}         # id -> [color, shape, x, y, size, seen_ms]
        self.last_target = None

    def see(self, oid, color, shape, x, y, size, now):
        new = oid not in self.objects
        self.objects[oid] = [color, shape, x, y, size, now]
        return new

    def label(self, oid):
        o = self.objects[oid]
        return "%s %s #%d" % (o[0], o[1], oid)

    def visible(self, now):
        return [i for i, o in self.objects.items() if time.ticks_diff(now, o[5]) < SEEN_MS]

    def resolve(self, desc, pose, now, visible_only=False):
        """'blue square' / 'red pillar' / '#3' / 'it' -> object id, or None."""
        words = desc.lower().replace("#", " #").split()
        for w in words:
            if w.startswith("#") and w[1:].isdigit() and int(w[1:]) in self.objects:
                return int(w[1:])
        color = None
        shape = None
        for w in words:
            if w in COLORS:
                color = w
            for s, names in SHAPES.items():
                if w in names.split():
                    shape = s
        if color is None and shape is None:
            if self.last_target in self.objects:
                return self.last_target
        x, y, _ = pose
        best, best_key = None, None
        for i, o in self.objects.items():
            if (color and o[0] != color) or (shape and o[1] != shape):
                continue
            in_view = time.ticks_diff(now, o[5]) < SEEN_MS
            if visible_only and not in_view:
                continue
            key = (0 if in_view else 1, hypot(o[2] - x, o[3] - y))  # what it can see now, then nearest
            if best_key is None or key < best_key:
                best, best_key = i, key
        return best


def exact(text):
    """'go 2; approach blue box' or 'go(2)' -> calls, or None if any part isn't a menu call."""
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
        if sk.params and sk.params[0][0] in ("text", "object"):
            args = [rest.strip(" \"'") or "it"]
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

    def __init__(self, post, memory):
        self.post = post
        self.mem = memory
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
        x, y, h = pose
        mem = self.mem
        if name == "stop":
            self.stop()
            self.post("stopped")
            return None
        if name == "face":
            face.set(mood=args[0] or "happy")
            self.post("face " + (args[0] or "happy"))
            return None
        if name == "say":
            face.set(caption=args[0])
            self.post("showing: " + args[0][:40])
            return None
        if name == "status":
            hx, hy, _ = mem.home or pose
            self.post("at %.1f, %.1f facing %d deg (started at %.1f, %.1f); %s; know %d objects" % (
                x, y, math.degrees(h), hx, hy, self.status, len(mem.objects)))
            return None
        if name == "look":
            seen = mem.visible(now)
            if not seen:
                self.post("I can't see any objects from here")
                return None
            parts = []
            for i in sorted(seen, key=lambda i: hypot(mem.objects[i][2] - x, mem.objects[i][3] - y)):
                o = mem.objects[i]
                b = math.degrees(wrap(math.atan2(o[3] - y, o[2] - x) - h))
                parts.append("%s %.1fm %s" % (mem.label(i), hypot(o[2] - x, o[3] - y), direction(b)))
            self.post("I see: " + "; ".join(parts))
            return None

        defaults = {"go": 1.0, "turn": 90.0, "explore": 20.0}
        if name in defaults and (not args or args[0] is None):
            args = [defaults[name]]
        c = {"name": name, "args": args, "pose": pose, "t": now}
        if name == "go" and abs(args[0]) > MAX_GO:
            self.post("%s fail too far" % fmt((name, args)))
            return None
        if name == "goto" and hypot(args[0] - x, args[1] - y) > MAX_GOTO:
            self.post("%s fail too far" % fmt((name, args)))
            return None
        if name == "approach":
            oid = mem.resolve(args[0], pose, now)
            if oid is None:
                self.post("approach fail: I haven't seen a %s yet (try: find %s)" % (args[0], args[0]))
                return None
            mem.last_target = oid
            c["target"] = oid
            self.status = "approaching " + mem.label(oid)
        elif name == "find":
            c["turned"], c["last"] = 0.0, h
            self.status = "looking for " + args[0]
        elif name == "return":
            if mem.home is None:
                self.post("return fail: no start recorded")
                return None
            self.status = "returning to start"
        else:
            self.status = fmt((name, args))
        face.set(caption=self.status[:16])
        return c

    def _finish(self, ok, why=""):
        if ok and why:
            self.post(why)
        else:
            self.post(self.cur["name"] + (" done" if ok else " fail " + why))
        self.cur = None
        self.status = "idle" if not self.queue else self.status

    def _drive_to(self, tx, ty, ranges, pose, arrive=0.3, avoid=True):
        """Steer towards (tx, ty), pushed away from whatever the range sensors see.
        Returns wheel speeds, or None once within `arrive` metres."""
        x, y, h = pose
        dx, dy = tx - x, ty - y
        if hypot(dx, dy) < arrive:
            return None
        c = self.cur
        now = time.ticks_ms()
        if avoid and time.ticks_diff(c.get("escape_until", now), now) > 0:
            # escaping: commit to going round on the side that had more room
            w = 0.5 * c["escape_side"]
            if min(ranges[1:4]) < 0.35:
                return -w, w                        # turn away until the front is clear
            return 0.5 - 0.15 * c["escape_side"], 0.5 + 0.15 * c["escape_side"]
        wheels = self._steer(err=wrap(math.atan2(dy, dx) - h), ranges=ranges, h=h, bearing=math.atan2(dy, dx), avoid=avoid)
        if avoid and abs(wheels[0]) + abs(wheels[1]) < 0.1:
            # pull towards the target and push from an obstacle cancel out: after a second, escape
            c["stall"] = c.get("stall", 0) + 1
            if c["stall"] > 10:
                c["stall"] = 0
                c["escape_side"] = 1 if ranges[3] + ranges[4] >= ranges[0] + ranges[1] else -1
                c["escape_until"] = time.ticks_add(now, 2500)
        else:
            c["stall"] = 0
        return wheels

    def _steer(self, err, ranges, h, bearing, avoid):
        if avoid and min(ranges[1:4]) < 0.2:  # something right in front: back off first
            return -0.4, -0.4
        if abs(err) > 0.8 and (not avoid or min(ranges) > 0.3):
            # target well off to the side and room to spin: turn on the spot, don't arc into things
            return self._face_towards(bearing, h) or (0.0, 0.0)
        steer = 1.2 * err
        fwd = 0.7 * clamp(1 - abs(err) / 1.2, 0, 1)
        if avoid:
            # each sensor that sees something within ~1 m pushes the heading away from it;
            # straight ahead pushes towards whichever side has more room
            push = 0.0
            for a, d in zip(SENSOR_ANGLES, ranges):
                if d < 0.7:
                    side = -1 if a > 0 else 1 if a < 0 else (1 if ranges[3] + ranges[4] > ranges[0] + ranges[1] else -1)
                    push += (0.7 - d) * side
            steer += 2.0 * push
            fwd *= clamp(min(ranges[1:4]) / 0.6, 0.3, 1)  # slow down when something is ahead
        turn = clamp(steer, -0.7, 0.7)
        return fwd - turn, fwd + turn

    def _face_towards(self, bearing, h):
        """Turn on the spot to face `bearing`. Returns wheel speeds, or None once facing it."""
        err = wrap(bearing - h)
        if abs(err) < math.radians(5):
            return None
        w = clamp(1.5 * err, -0.6, 0.6)
        w = w if abs(w) > 0.3 else (0.3 if w > 0 else -0.3)
        return -w, w

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
        name, args = c["name"], c["args"]
        mem = self.mem

        if name == "go":
            d = args[0]
            sign = 1 if d >= 0 else -1
            remaining = abs(d) - hypot(x - x0, y - y0)
            if remaining <= 0.03:
                self._finish(True)
                return 0.0, 0.0
            if sign > 0 and min(ranges[1:4]) < 0.2:
                self._finish(False, "blocked")
                return 0.0, 0.0
            if elapsed > abs(d) * 3 + 5:
                self._finish(False, "stuck")
                return 0.0, 0.0
            speed = sign * clamp(remaining / 0.6 + 0.15, 0.15, 0.7)
            turn = clamp(2.0 * wrap(h0 - h), -0.4, 0.4)
            return speed - turn, speed + turn

        if name in ("turn", "find"):
            c.setdefault("turned", 0.0)
            c["turned"] += wrap(h - c.get("last", h0))
            c["last"] = h
            if name == "find":
                oid = mem.resolve(args[0], pose, now, visible_only=True)
                if oid is not None:
                    mem.last_target = oid
                    o = mem.objects[oid]
                    self._finish(True, "found %s at %.1f, %.1f, %.1f m away" % (
                        mem.label(oid), o[2], o[3], hypot(o[2] - x, o[3] - y)))
                    return 0.0, 0.0
                if c["turned"] > math.radians(370):
                    self._finish(False, "can't see a %s anywhere from here" % args[0])
                    return 0.0, 0.0
                return -0.45, 0.45
            err = math.radians(args[0]) - c["turned"]
            if abs(err) < math.radians(3):
                self._finish(True)
                return 0.0, 0.0
            if elapsed > 20:
                self._finish(False, "stuck")
                return 0.0, 0.0
            w = clamp(1.5 * err, -0.8, 0.8)
            w = w if abs(w) > 0.3 else (0.3 if w > 0 else -0.3)
            return -w, w

        if name == "approach":
            o = mem.objects[c["target"]]
            dist = hypot(o[2] - x, o[3] - y)
            gap = dist - o[4] - 0.55  # robot's half-length + the object's half-size
            if elapsed > 90:
                self._finish(False, "timeout")
                return 0.0, 0.0
            if gap > 0.4:
                return self._drive_to(o[2], o[3], ranges, pose, arrive=0, avoid=gap > 1.2)
            turning = self._face_towards(math.atan2(o[3] - y, o[2] - x), h)
            if turning:
                return turning
            face.set(mood="happy", caption=(o[0] + " " + o[1])[:16])
            self._finish(True, "at the %s: a %s %s about %.1f m across, at %.1f, %.1f" % (
                mem.label(c["target"]), o[0], o[1], o[4] * 2, o[2], o[3]))
            return 0.0, 0.0

        if name == "return":
            hx, hy, hh = mem.home
            if not c.get("arrived"):
                wheels = self._drive_to(hx, hy, ranges, pose)
                if wheels:
                    return wheels if elapsed < 120 else self._fail_stop("timeout")
                c["arrived"] = True
            turning = self._face_towards(hh, h)
            if turning:
                return turning
            self._finish(True, "back at the start")
            return 0.0, 0.0

        if name == "goto":
            if elapsed > 120:
                return self._fail_stop("timeout")
            wheels = self._drive_to(args[0], args[1], ranges, pose)
            if wheels is None:
                self._finish(True)
                return 0.0, 0.0
            return wheels

        if name == "explore":
            if elapsed > args[0]:
                self._finish(True)
                return 0.0, 0.0
            return act(weights, ranges)

        return self._fail_stop("unknown skill")

    def _fail_stop(self, why):
        self._finish(False, why)
        return 0.0, 0.0


def mood_for(ranges):
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
nlu = Understander(Embeddings("nlu_hash.bin", "nlu_vec.bin", "nlu_rank.bin", 64), MENU, object_words=OBJECT_WORDS)
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

    mem = Memory()
    skills = Skills(post, mem)
    micropython.kbd_intr(-1)  # 0x03 bytes are data now, not Ctrl-C
    out.write(XT.pack_register(NAME))

    now = time.ticks_ms()
    last_data = time.ticks_add(now, -IDLE_MS)
    last_status = now
    last_face = now
    held_until = now
    menu_posted = [time.ticks_add(now, -60000)]

    def instruct(sender, text):
        calls = exact(text)
        how = "exact"
        if sender == PILOT:
            if calls is None and ";" in text:
                calls = exact(text.rsplit(";", 1)[0])  # long plans get cut off mid-call; drop the stub
            if calls is None:
                # the pilot must answer with exact calls; never guess at machine-written plans
                post("can't run pilot plan: " + text[:60])
                return
            calls = calls[:12]  # small models sometimes repeat a pattern until they run out of tokens
            how = "pilot"
        if calls is None:
            face.set(caption="thinking...")
            face.draw()
            calls, conf = nlu.parse(text)
            how = "understood %.2f" % conf
        if calls is None:
            if time.ticks_diff(time.ticks_ms(), menu_posted[0]) > 60000:
                for line in MENU_POST:
                    post(line)
                menu_posted[0] = time.ticks_ms()
            post("not sure what that means; asking %s" % PILOT)
            out.write(XT.pack_dm(PILOT, "%s asks %s: %s" % (sender, NAME, text[:70]), True))
            face.set(mood="alert", caption="asking pilot")
            return
        if calls[0][0] == "stop" or any(n not in ("face", "say", "status", "look") for n, _ in calls):
            skills.stop("replanned")  # a new movement order replaces the old plan
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
                now = time.ticks_ms()
                if len(parts) == 7 and parts[0] == "o":
                    try:
                        oid = int(parts[1])
                        ox, oy, size = int(parts[4]) / 100, int(parts[5]) / 100, int(parts[6]) / 100  # size = radius
                    except ValueError:
                        continue
                    if mem.see(oid, parts[2], parts[3], ox, oy, size, now):
                        post("%s sees: %s at %.1f, %.1f" % (NAME, mem.label(oid), ox, oy))
                    continue
                if len(parts) != 10 or parts[0] != "s":
                    continue
                try:
                    ranges = [int(p) / 100 for p in parts[2:7]]
                    pose = (int(parts[7]) / 100, int(parts[8]) / 100, math.radians(int(parts[9])))
                except ValueError:
                    continue
                if mem.home is None:
                    mem.home = pose
                try:
                    left, right = skills.step(ranges, pose, now)
                except Exception as err:  # report it and stop safely instead of dying
                    skills.stop("error")
                    post("error in skill: %r" % err)
                    left = right = 0.0
                out.write(XT.pack_dm(SIM, "m %s %d %d" % (parts[1], int(left * 100), int(right * 100)), False))
                if time.ticks_diff(now, held_until) > 0:
                    face.set(mood=mood_for(ranges), look=(left - right) * 1.5,
                             caption=skills.status[:16] if skills.cur else "ready")
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
            elif low.startswith("@") or sender in (NAME, SIM, "bridge"):
                continue  # addressed to someone else, or another robot's own chatter
            # anything else a person or agent says in the channel is an instruction for this robot
            if text in ("skills", "help", "?"):
                for line in MENU_POST:
                    post(line)
                menu_posted[0] = time.ticks_ms()
                continue
            try:
                instruct(sender, text)
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
