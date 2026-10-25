# Upload the brain to the ESP32 and restart it.
#   .venv/bin/python deploy.py [--port /dev/cu.usbserial-210]
# The running brain ignores Ctrl-C, so this first DMs it "!repl" to hand the REPL back.
# mpremote skips files that are already up to date, so only changed files are sent.
import argparse
import os
import struct
import subprocess
import sys
import time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "esp32/main.py": "main.py",
    "esp32/face.py": "face.py",
    "esp32/crosstalk_micro.py": "crosstalk_micro.py",
    "esp32/weights.json": "weights.json",
    "esp32/nlu_hash.bin": "nlu_hash.bin",
    "esp32/nlu_rank.bin": "nlu_rank.bin",
    "esp32/nlu_vec.bin": "nlu_vec.bin",   # ~400 KB: about 3 minutes when it changes
    "shared/policy.py": "policy.py",
    "shared/nlu.py": "nlu.py",
    "shared/husky_menu.py": "husky_menu.py",
}


def stop_brain(port):
    payload = b"\x06deploy!repl"
    frame = struct.pack(">BBBBIH", 0x58, 1, 0x03, 0, int(time.time()), len(payload)) + payload
    with serial.Serial(port, 115200, timeout=0) as s:
        s.write(struct.pack(">H", len(frame)) + frame)
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-210")
    args = ap.parse_args()

    stop_brain(args.port)
    # Run mpremote through this Python: script #! lines break on paths with spaces.
    mpremote = os.path.join(os.path.dirname(sys.executable), "mpremote")
    # "resume" skips mpremote's soft reset, which would start the brain again before the copy
    cmd = [sys.executable, mpremote, "connect", args.port, "resume"]
    for src, dst in FILES.items():
        cmd += ["cp", os.path.join(HERE, src), ":" + dst, "+"]
    cmd += ["reset"]
    subprocess.run(cmd, check=True)
    print("deployed; the brain restarts on its own")


if __name__ == "__main__":
    main()
