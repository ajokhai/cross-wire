# Minimal Python stand-in for crosstalk-bridge: puts the ESP32 into a Cross Talk channel
# as one agent, passing frames between USB serial and the hub.
#
#   .venv/bin/python bridge/xt_bridge.py --channel xt_... [--port /dev/cu.usbserial-210] [--name husky]
#
# Device -> channel: BROADCAST -> message, DIRECT_MSG -> DM, HEARTBEAT -> status.
# Channel -> device: messages and DMs as [from_len][from][text], cut to fit a 128-byte frame.
# Local agents (like the sim, which shares this serial port) can talk to the device directly
# with send_local(); the device's DMs to them land in local_inbox instead of going to the hub.
import argparse
import os
import queue
import struct
import sys
import threading
import time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, os.path.join(HERE, "..", "esp32"))

from crosstalk import CrossTalk, CrossTalkError  # noqa: E402
from crosstalk_micro import FrameReader, OP_BROADCAST, OP_DIRECT_MSG, OP_HEARTBEAT, XT_HEADER_SIZE  # noqa: E402

DEVICE_MAX_FRAME = 240  # the brain reads frames up to 256 bytes; keep a little margin


def frame(opcode, payload):
    body = struct.pack(">BBBBIH", 0x58, 1, opcode, 0, int(time.time()) & 0xFFFFFFFF, len(payload)) + payload
    return struct.pack(">H", len(body)) + body


def attributed(sender, text):
    who = sender.encode()[:32]
    room = DEVICE_MAX_FRAME - XT_HEADER_SIZE - 1 - len(who)
    return bytes([len(who)]) + who + text.encode()[:room]


class DeviceBridge:
    def __init__(self, port, channel, name, url, local_names=(), log=print):
        self.port = serial.Serial(port, 115200, timeout=0.02)
        self.write_lock = threading.Lock()
        self.local_names = set(local_names)
        self.local_inbox = queue.Queue()
        self.log = log
        self.status = ""
        self.ct = None
        if channel:
            self.ct = CrossTalk.connect(name, url=url, environment="device")
            self.channel = self.ct.join_channel(channel).address
            self.me = self.ct.agent["id"]
            self.ct.on("message", lambda m: self._to_device("message", m))
            self.ct.on("dm", lambda m: self._to_device("dm", m))
            log("bridge: %s joined %s" % (name, self.channel))
        self.reader = FrameReader(max_frame=256)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _write(self, data):
        with self.write_lock:
            self.port.write(data)

    def send_local(self, sender, text, dm=True):
        """Talk to the device as a local agent (no hub involved)."""
        self._write(frame(OP_DIRECT_MSG if dm else OP_BROADCAST, attributed(sender, text)))

    def _to_device(self, kind, message):
        sender = message.get("from", {})
        if sender.get("id") == self.me:
            return
        content = message.get("content", "")
        self._write(frame(OP_DIRECT_MSG if kind == "dm" else OP_BROADCAST, attributed(sender.get("name", "?"), content)))
        self.log("-> device %s %s: %s" % (kind, sender.get("name"), content[:80]))

    def _read_loop(self):
        last_status = None
        while True:
            data = self.port.read(512)
            for f in self.reader.feed(data) if data else ():
                p = bytes(f["payload"])
                try:
                    if f["opcode"] == OP_DIRECT_MSG and p and len(p) > p[0]:
                        to, text = p[1:1 + p[0]].decode(), p[1 + p[0]:].decode("utf-8", "replace")
                        if to in self.local_names:
                            self.local_inbox.put(text)
                        elif self.ct:
                            self.ct.dm(to, text, reply_expected=bool(f["flags"] & 1))
                            self.log("<- device dm %s: %s" % (to, text))
                    elif f["opcode"] == OP_BROADCAST and p:
                        text = p.decode("utf-8", "replace")
                        self.log("<- device says: %s" % text)
                        if self.ct:
                            self.ct.send(self.channel, text)
                    elif f["opcode"] == OP_HEARTBEAT:
                        self.status = p.decode("utf-8", "replace")
                        if self.ct and self.status != last_status:
                            self.ct.set_status("working", current_task=self.status)
                            last_status = self.status
                except CrossTalkError as err:
                    self.log("hub refused: %s" % err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True)
    ap.add_argument("--port", default="/dev/cu.usbserial-210")
    ap.add_argument("--name", default="husky")
    ap.add_argument("--url", default=os.environ.get("CROSSTALK_URL", "ws://localhost:4488"))
    args = ap.parse_args()
    bridge = DeviceBridge(args.port, args.channel, args.name, args.url)
    bridge.send_local("bridge", "@%s skills" % args.name, dm=False)  # ask the device to introduce itself
    try:
        while bridge.ct.connected:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
