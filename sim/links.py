# Ways for the sim to reach the robot brain. Each link has send(text) and
# recv(timeout) -> text or None.
#   SerialLink    talks Cross Talk binary frames straight to the ESP32 (no hub), acting as the bridge would.
#   CrossTalkLink DMs the brain through a Cross Talk hub (the ESP32 joins via crosstalk-bridge).
#   LocalLink     runs the same policy on the Mac, for comparison.
import os
import queue
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
sys.path.insert(0, os.path.join(HERE, "..", "esp32"))

OP_DIRECT_MSG = 0x03


class SerialLink:
    name = "serial"

    def __init__(self, port, baud=115200, me="robot-sim"):
        import serial
        from crosstalk_micro import FrameReader

        self.me = me.encode()
        self.port = serial.Serial(port, baud, timeout=0)
        self.reader = FrameReader(max_frame=256)
        self.pending = []
        self.brain_status = ""

    def send(self, text):
        payload = bytes([len(self.me)]) + self.me + text.encode()
        frame = struct.pack(">BBBBIH", 0x58, 1, OP_DIRECT_MSG, 0, int(time.time()) & 0xFFFFFFFF, len(payload)) + payload
        self.port.write(struct.pack(">H", len(frame)) + frame)

    def recv(self, timeout):
        end = time.monotonic() + timeout
        while True:
            if self.pending:
                return self.pending.pop(0)
            data = self.port.read(512)
            for f in self.reader.feed(data) if data else ():
                p = bytes(f["payload"])
                if f["opcode"] == OP_DIRECT_MSG and p and len(p) > p[0]:
                    self.pending.append(p[1 + p[0]:].decode("utf-8", "replace"))
                elif f["opcode"] == 0x00:  # heartbeat carries the brain's status
                    self.brain_status = p.decode("utf-8", "replace")
            if not self.pending:
                if time.monotonic() >= end:
                    return None
                time.sleep(0.002)


class CrossTalkLink:
    name = "crosstalk"

    def __init__(self, channel, brain="esp32-brain", url="ws://localhost:4488", me="robot-sim"):
        from crosstalk import CrossTalk

        self.brain = brain
        self.inbox = queue.Queue()
        self.brain_status = ""
        self.ct = CrossTalk.connect(me, url=url)
        self.ct.on("dm", self._on_dm)
        self.ct.on("member.updated", self._on_member)
        if channel:
            self.channel = self.ct.join_channel(channel)
        else:
            self.channel = self.ct.create_channel("robot-sim", topic="2D rover driven by an ESP32 brain")
        print("Cross Talk channel:", self.channel.address)

    def _on_dm(self, message):
        if message.get("from", {}).get("name") == self.brain:
            self.inbox.put(message.get("content", ""))

    def _on_member(self, agent, _channel):
        if agent.get("name") == self.brain:
            self.brain_status = agent.get("currentTask") or agent.get("status") or ""

    def send(self, text):
        self.ct.dm(self.brain, text, reply_expected=False)

    def recv(self, timeout):
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None


class LocalLink:
    name = "local"

    def __init__(self):
        import json
        from policy import act

        with open(os.path.join(HERE, "..", "esp32", "weights.json")) as f:
            self.weights = json.load(f)
        self.act = act
        self.reply = None
        self.brain_status = "running on the Mac"

    def send(self, text):
        parts = text.split()
        left, right = self.act(self.weights, [int(p) / 100 for p in parts[2:]])
        self.reply = "m %s %d %d" % (parts[1], int(left * 100), int(right * 100))

    def recv(self, timeout):
        reply, self.reply = self.reply, None
        return reply
