# OLED demo for ESP32 + GME12864-43 (SSD1306, 128x64, I2C 0x3c)
# Wiring: SDA -> GPIO21, SCL -> GPIO22
import time, random
from machine import I2C, Pin
import ssd1306

W, H = 128, 64
oled = ssd1306.SSD1306_I2C(W, H, I2C(0, sda=Pin(21), scl=Pin(22), freq=400000))


def title(msg):
    oled.fill(0)
    oled.text(msg, (W - len(msg) * 8) // 2, 28, 1)
    oled.show()
    time.sleep(1)


def starfield(seconds=8):
    stars = [[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.1, 1)] for _ in range(60)]
    end = time.ticks_add(time.ticks_ms(), seconds * 1000)
    while time.ticks_diff(end, time.ticks_ms()) > 0:
        oled.fill(0)
        for s in stars:
            s[2] -= 0.02
            if s[2] <= 0.02:
                s[0], s[1], s[2] = random.uniform(-1, 1), random.uniform(-1, 1), 1
            x = int(W / 2 + s[0] / s[2] * 32)
            y = int(H / 2 + s[1] / s[2] * 32)
            if 0 <= x < W and 0 <= y < H:
                oled.pixel(x, y, 1)
                if s[2] < 0.3:  # close stars look bigger
                    oled.fill_rect(x, y, 2, 2, 1)
        oled.show()


def bouncing(seconds=8):
    balls = [[random.randint(4, W - 8), random.randint(4, H - 8),
              random.choice((-2, -1, 1, 2)), random.choice((-2, -1, 1, 2)),
              random.randint(2, 4)] for _ in range(5)]
    end = time.ticks_add(time.ticks_ms(), seconds * 1000)
    while time.ticks_diff(end, time.ticks_ms()) > 0:
        oled.fill(0)
        oled.rect(0, 0, W, H, 1)
        for b in balls:
            b[0] += b[2]
            b[1] += b[3]
            r = b[4]
            if b[0] - r <= 0 or b[0] + r >= W - 1:
                b[2] = -b[2]
            if b[1] - r <= 0 or b[1] + r >= H - 1:
                b[3] = -b[3]
            oled.ellipse(b[0], b[1], r, r, 1, True)
        oled.show()


def life(seconds=10):
    # Game of Life on a 64x32 grid, each cell drawn as 2x2 pixels
    cw, ch = W // 2, H // 2
    grid = bytearray(random.getrandbits(1) for _ in range(cw * ch))
    end = time.ticks_add(time.ticks_ms(), seconds * 1000)
    while time.ticks_diff(end, time.ticks_ms()) > 0:
        oled.fill(0)
        for y in range(ch):
            row = y * cw
            for x in range(cw):
                if grid[row + x]:
                    oled.fill_rect(x * 2, y * 2, 2, 2, 1)
        oled.show()
        nxt = bytearray(cw * ch)
        for y in range(ch):
            up, dn = ((y - 1) % ch) * cw, ((y + 1) % ch) * cw
            row = y * cw
            for x in range(cw):
                l, r = (x - 1) % cw, (x + 1) % cw
                n = (grid[up + l] + grid[up + x] + grid[up + r] +
                     grid[row + l] + grid[row + r] +
                     grid[dn + l] + grid[dn + x] + grid[dn + r])
                if n == 3 or (n == 2 and grid[row + x]):
                    nxt[row + x] = 1
        grid = nxt


while True:
    title("Starfield")
    starfield()
    title("Bounce")
    bouncing()
    title("Game of Life")
    life()
