# Runs the rover sim at 10 Hz, asks the brain what to do each step, and serves a
# live view at http://localhost:8765.
#
#   python sim/sim.py --link serial --port /dev/cu.usbserial-210   # ESP32 directly over USB
#   python sim/sim.py --link crosstalk [--channel xt_...]          # ESP32 through a Cross Talk hub
#   python sim/sim.py --link local                                 # same net, on the Mac
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from links import CrossTalkLink, LocalLink, SerialLink
from world import DT, World

HERE = os.path.dirname(os.path.abspath(__file__))
REPLY_TIMEOUT = 0.25


class Sim:
    def __init__(self, link, seed):
        self.link = link
        self.lock = threading.Lock()
        self.world = World(seed=seed)
        self.seq = 0
        self.answered = 0
        self.missed = 0
        self.latency_ms = 0.0
        self.reset_requested = False

    def step(self):
        w = self.world
        readings = [int(round(s * 100)) for s in w.sensors()]
        self.seq += 1
        sent = time.monotonic()
        self.link.send("s %d %s" % (self.seq, " ".join(map(str, readings))))

        command = None
        while command is None:
            remaining = REPLY_TIMEOUT - (time.monotonic() - sent)
            reply = self.link.recv(max(0.0, remaining))
            if reply is None:
                break
            parts = reply.split()
            if len(parts) == 4 and parts[0] == "m" and parts[1] == str(self.seq):
                command = (int(parts[2]) / 100, int(parts[3]) / 100)
            # anything else is a stale or unrelated reply; keep waiting

        with self.lock:
            if command is None:
                self.missed += 1
                w.step(0, 0)  # no answer: stop rather than drive blind
            else:
                self.answered += 1
                self.latency_ms = 0.8 * self.latency_ms + 0.2 * (time.monotonic() - sent) * 1000
                w.step(*command)
            if self.reset_requested:
                self.world = World()
                self.reset_requested = False

    def snapshot(self):
        with self.lock:
            s = self.world.state()
            s.update(link=self.link.name, steps=self.seq, answered=self.answered, missed=self.missed,
                     latency_ms=round(self.latency_ms, 1), brain_status=self.link.brain_status,
                     sensors=[round(v, 2) for v in self.world.sensors()])
            return s


def serve(sim, port):
    page = open(os.path.join(HERE, "viewer.html"), "rb").read()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/state":
                self._send(json.dumps(sim.snapshot()).encode(), "application/json")
            else:
                self._send(page, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path == "/reset":
                sim.reset_requested = True
            self._send(b"{}", "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Viewer: http://localhost:%d" % port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", choices=["serial", "crosstalk", "local"], default="serial")
    ap.add_argument("--port", default="/dev/cu.usbserial-210", help="ESP32 serial port (serial link)")
    ap.add_argument("--channel", help="Cross Talk channel to join; omit to create one")
    ap.add_argument("--brain", default="esp32-brain", help="brain's agent name in the channel")
    ap.add_argument("--hub", default=os.environ.get("CROSSTALK_URL", "ws://localhost:4488"))
    ap.add_argument("--http", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=0, help="stop after N steps (0 = run forever)")
    args = ap.parse_args()

    if args.link == "serial":
        link = SerialLink(args.port)
    elif args.link == "crosstalk":
        link = CrossTalkLink(args.channel, brain=args.brain, url=args.hub)
    else:
        link = LocalLink()

    sim = Sim(link, args.seed)
    if args.http:
        serve(sim, args.http)
    next_tick = time.monotonic()
    try:
        while not args.steps or sim.seq < args.steps:
            sim.step()
            if sim.seq % 50 == 0:
                s = sim.snapshot()
                print("step %d  answered %d  missed %d  latency %.1f ms  distance %.1f m  bumps %d  [%s]" % (
                    s["steps"], s["answered"], s["missed"], s["latency_ms"], s["distance"],
                    s["collisions"], s["brain_status"]), flush=True)
            next_tick += DT
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
