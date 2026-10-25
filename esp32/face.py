# Animated robot face for a 128x64 SSD1306.
# Moods: "happy" (cruising), "alert" (something close), "scared" (about to hit),
# "sleepy" (no sensor data), "boot".
import random
import time

EYE_W, EYE_H = 30, 28
EYE_Y = 6
LEFT_X, RIGHT_X = 22, 76


def _approach(cur, target, rate):
    return cur + (target - cur) * rate


class Face:
    def __init__(self, oled):
        self.oled = oled
        self.mood = "boot"
        self.look = 0.0          # -1 = looking left, 1 = right
        self._dx = 0.0           # smoothed eye offset
        self._h = float(EYE_H)   # smoothed eye height
        self._next_blink = time.ticks_add(time.ticks_ms(), 1500)
        self._blink_until = 0
        self.caption = ""

    def set(self, mood=None, look=None, caption=None):
        if mood is not None:
            self.mood = mood
        if look is not None:
            self.look = max(-1.0, min(1.0, look))
        if caption is not None:
            self.caption = caption

    def _eye(self, x, y, w, h):
        o = self.oled
        if h <= 3:
            o.fill_rect(x, y + EYE_H // 2 - 1, w, 3, 1)
            return
        top = y + (EYE_H - h) // 2
        o.fill_rect(x, top, w, h, 1)
        # rounded corners
        for cx, cy in ((x, top), (x + w - 1, top), (x, top + h - 1), (x + w - 1, top + h - 1)):
            o.pixel(cx, cy, 0)
        o.pixel(x + 1, top, 0); o.pixel(x, top + 1, 0)
        o.pixel(x + w - 2, top, 0); o.pixel(x + w - 1, top + 1, 0)
        o.pixel(x + 1, top + h - 1, 0); o.pixel(x, top + h - 2, 0)
        o.pixel(x + w - 2, top + h - 1, 0); o.pixel(x + w - 1, top + h - 2, 0)

    def draw(self):
        o = self.oled
        now = time.ticks_ms()
        mood = self.mood

        target_h = {"happy": EYE_H, "alert": EYE_H, "scared": EYE_H + 6,
                    "sleepy": 6, "boot": EYE_H}.get(mood, EYE_H)
        target_w = EYE_W + (4 if mood == "scared" else 0)

        # blinking (not while scared: eyes stay wide open)
        if mood != "scared" and time.ticks_diff(now, self._next_blink) >= 0:
            self._blink_until = time.ticks_add(now, 140)
            self._next_blink = time.ticks_add(now, random.randint(1800, 5000))
        if time.ticks_diff(self._blink_until, now) > 0:
            target_h = 2

        rate = 0.6 if target_h == 2 else 0.35
        self._h = _approach(self._h, target_h, rate)
        self._dx = _approach(self._dx, self.look * 12, 0.3)
        h = int(self._h + 0.5)
        dx = int(self._dx)
        jitter = random.randint(-1, 1) if mood == "scared" else 0

        o.fill(0)
        self._eye(LEFT_X + dx - (target_w - EYE_W) // 2 + jitter, EYE_Y, target_w, h)
        self._eye(RIGHT_X + dx - (target_w - EYE_W) // 2 + jitter, EYE_Y, target_w, h)

        # happy: cheeks squint the bottom of the eyes
        if mood == "happy" and h > 10:
            o.ellipse(LEFT_X + dx + EYE_W // 2, EYE_Y + EYE_H + 6, EYE_W // 2 + 4, 9, 0, True)
            o.ellipse(RIGHT_X + dx + EYE_W // 2, EYE_Y + EYE_H + 6, EYE_W // 2 + 4, 9, 0, True)

        # mouth
        mx, my = 64 + dx // 2, 46
        if mood == "happy":
            o.ellipse(mx, my - 2, 9, 5, 1, False, 0b1100)   # smile (bottom half)
        elif mood == "alert":
            o.hline(mx - 6, my, 12, 1)
        elif mood == "scared":
            o.ellipse(mx, my, 5, 4, 1, True)
        elif mood == "sleepy":
            o.hline(mx - 4, my, 8, 1)
            z = (now // 600) % 3
            o.text("z" * (z + 1), 100, 4, 1)
        else:  # boot
            o.hline(mx - 4, my, 8, 1)

        if self.caption:
            o.text(self.caption[:16], max(0, 64 - len(self.caption[:16]) * 4), 56, 1)
        o.show()
