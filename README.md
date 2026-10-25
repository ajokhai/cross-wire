# ESP32 robot brain

An ESP32 (MicroPython) runs a tiny neural net that drives a simulated rover and shows
an animated face on a 128x64 OLED (GME12864-43, SSD1306, I2C 0x3c, SDA=GPIO21, SCL=GPIO22).
The sim runs on the Mac and talks to the brain over USB serial, directly or through a
[Cross Talk](https://github.com/ajokhai/cross-talk) hub.

```
sim/sim.py ──"s <seq> d1..d5"──▶ ESP32 brain (esp32/main.py) ──▶ policy net (shared/policy.py)
           ◀──"m <seq> L R"────                                └─▶ face on the OLED (esp32/face.py)
```

| Path | What it is |
| :-- | :-- |
| `shared/policy.py` | The net: 5 range sensors -> 6 hidden -> 2 wheel speeds, 50 weights. Same file runs on the Mac and the ESP32. |
| `sim/world.py` | 2D differential-drive rover, walled arena, round obstacles, 5 range sensors. |
| `sim/train.py` | Evolves the weights in the headless sim and writes `esp32/weights.json` (~15 s). |
| `sim/sim.py` | Runs the sim at 10 Hz against a brain; live view at http://localhost:8765. |
| `sim/links.py` | How the sim reaches the brain: `serial`, `crosstalk` or `local`. |
| `esp32/` | What runs on the board: brain (`main.py`), face, Cross Talk micro client, weights. |
| `deploy.py` | Uploads `esp32/` + `shared/policy.py` to the board and restarts it. |

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install pyserial mpremote esptool websocket-client
```

## Run

```sh
.venv/bin/python deploy.py                          # after changing anything in esp32/ or shared/
.venv/bin/python sim/sim.py --link serial           # ESP32 over USB, no hub
.venv/bin/python sim/sim.py --link local            # same net on the Mac, for comparison
.venv/bin/python sim/train.py                       # retrain, then deploy
```

### Through Cross Talk

The hub needs to allow 10 messages a second per agent, so run it without rate limits
(it only listens on localhost): `crosstalk serve --no-rate-limit`. Then:

```sh
.venv/bin/python sim/sim.py --link crosstalk        # creates a channel and prints xt_...
crosstalk-bridge --channel xt_... --serial /dev/cu.usbserial-210 --name esp32-brain
```

`crosstalk-bridge` comes from `packages/embedded` in the Cross Talk repo; it is not in the
published `cross-talk` package.

## Board notes

- While the brain runs, bytes from the Mac are data, so Ctrl-C does nothing. `deploy.py`
  sends the brain `!repl` first. Pressing the BOOT button also stops it.
- `esp32/demo_animations.py` is the earlier starfield / bounce / Game of Life demo.

## Benchmarks

`bench/` holds the tests behind the Cross Talk benchmark write-up; raw results are in
`bench/results/`. On 2026-10-25 the brain answered 600/600 steps over direct serial
(median round trip 48.9 ms) and 883/900 through a Cross Talk hub (median 51.9 ms, p99 87.5 ms).
