# ESP32 robot brain over Cross Talk

Plain-English instructions for a robot, with the decisions made on an ESP32:

> @husky turn around, investigate that blue square, then go back to where you started

The robot posts its skill menu and what it sees in a [Cross Talk](https://github.com/ajokhai/cross-talk)
channel; people and agents instruct it with `@husky ...`.

```
you / agents ──"@husky ..."──▶ #robot-lab (Cross Talk hub)
                                  │ bridge (USB serial)
ESP32 brain (esp32/main.py) ◀─────┘
 ├─ exact calls ("approach blue box; return")  run as they are
 ├─ plain English → on-chip command model      shared/nlu.py (word vectors, ~400 KB flash)
 ├─ not sure → DM @pilot                       a small LLM; must answer with exact calls
 ├─ memory                                     where it started; every object it has seen
 ├─ skills                                     go turn goto approach find return look explore ...
 └─ face on the OLED                           GME12864-43 (SSD1306, I2C 0x3c, SDA=21, SCL=22)
        ▲ ranges, pose, camera sightings / wheel commands, 10 Hz
MuJoCo sim (sim/mj_sim.py): Clearpath Husky URDF, coloured boxes and cylinders, a camera
```

The Mac only runs the world (physics, camera) and, for now, stands in for the onboard
computer that hosts the pilot (the tablet later).

## Start

```sh
./start.sh                                        # model server + channel + pilot; prints the channel
.venv/bin/mjpython sim/mj_sim.py --channel xt_... # robot + 3D view
crosstalk up xt_...                               # talk to it
```

Things to say:

| You say | It runs |
|---|---|
| what can you see? | `look`: lists objects in view, with distance and direction |
| turn around | `turn 180` |
| investigate that blue square | `approach blue box`: drives up to it, faces it, describes it |
| go back to where you started | `return` |
| find the red pillar | `find red cylinder`: spins until it sees one |
| check it out | `approach it` (the last object it found or visited) |
| where are you? | `status` |
| drive in a square with 1 m sides | asks the pilot, runs its plan |

## Files

| Path | What it is |
| :-- | :-- |
| `esp32/main.py` | The brain: instructions, memory, object resolution, skills, steering, face. |
| `shared/husky_menu.py` | The robot's self-description: skills, and the words its camera uses for things. |
| `shared/nlu.py` | On-chip command model: matches instructions to any skill menu by meaning. |
| `shared/policy.py` | Reflex net (5 ranges -> 2 wheels, 50 weights), used by `explore`. |
| `tools/build_nlu.py` | Builds the word vectors and scores the command model (UGV, arm and USV menus). |
| `sim/mj_world.py` | Any wheeled URDF in MuJoCo: wheels found automatically, range sensors, camera, objects. |
| `sim/mj_sim.py` | Runs the world, hosts the bridge, asks the brain for wheel commands at 10 Hz. |
| `bridge/xt_bridge.py` | Python stand-in for `crosstalk-bridge` (serial <-> hub). |
| `deploy.py` | Uploads the brain (stops the running one first; skips unchanged files). |
| `start.sh`, `pilot_prompt.txt` | Model server, channel and pilot. |
| `sim/world.py`, `sim/train.py` | 2D sim used to train the reflex net. |

Also in the repo: `bench/` (serial and hub round-trip benchmarks, measured on an earlier
firmware) and the original 2D rover sim (`sim/sim.py`, `sim/links.py`, `sim/viewer.html`),
which speaks an older sensor format than the current brain accepts.

## Known issues

- An instruction that starts with a skill name and continues ("find the red pillar, then
  check it out") is taken as one exact call, so the part after "then" is dropped.
- "drive in a square ..." is read as `approach square` ("square" is a word for a box).
- The 1.5B pilot sometimes echoes the robot's reply format; the robot then refuses its plan.
- `explore` still uses the reflex net trained in the 2D sim, which gets stuck on the Husky.

## Notes

- How it steers: towards the target, pushed away from whatever the range sensors see; if
  those cancel out for a second it commits to going round on the roomier side.
- Understanding an instruction takes ~0.5-1.5 s on the chip; the robot holds still meanwhile.
- The brain ignores Ctrl-C while running; `deploy.py` DMs it `!repl` first, or press BOOT.
- Sim and brain share one serial port, so only one of `mj_sim.py`, `xt_bridge.py`, `deploy.py` at a time.
- Safety: `go` over 10 m and `goto` over 20 m are refused; pilot plans run only as exact calls.
- Keep this folder out of iCloud-synced locations: evicted files make everything hang.
