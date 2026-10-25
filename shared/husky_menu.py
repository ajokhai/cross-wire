# The Husky's skill menu: what it tells the channel it can do, and what the on-chip
# command model matches instructions against. Shared by the ESP32 brain and tools/build_nlu.py.
NAME = "husky"

MENU = [
    "go(m) drive forward, or back up in reverse, a distance in meters",
    "turn(deg) turn, spin or rotate in place by an angle, left is positive",
    "goto(x,y) go to, navigate or head to a position x y on the map",
    "approach(object) investigate, inspect, check out or drive up to an object such as a colored box or cylinder",
    "find(object) search for, look for or locate an object such as a box or cylinder",
    "return() head back to the start, return to the place it started from",
    "look() describe what it can see",
    "explore(s) wander, roam and explore by itself while avoiding obstacles",
    "stop() stop moving, halt, freeze",
    "face(mood:happy|alert|scared|sleepy) show an emotion on its face",
    "say(text) show a message on its screen",
    "status() report where it is, its position and what it is doing",
]

# What its camera reports, and the everyday words people use for each (also part of the
# robot's self-description: words used for objects are arguments, not skill names).
COLORS = ("red", "blue", "green", "yellow", "orange", "purple")
SHAPES = {
    "box": "box boxes square cube crate block rectangle",
    "cylinder": "cylinder pillar post barrel column pole drum tube round circle can",
}
OBJECT_WORDS = set(COLORS) | {w for words in SHAPES.values() for w in words.split()} | {"object", "thing", "one"}

# What it posts in the channel (each line fits one Cross Talk frame)
MENU_POST = [
    NAME + " skills: go(m) turn(deg) goto(x,y) approach(object) find(object) return() look()",
    NAME + " skills: explore(s) stop() face(mood) say(text) status()",
    NAME + " args: m metres (- = back), deg (+ = left), x y metres, s seconds, object like 'blue box'",
]
