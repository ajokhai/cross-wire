# 2D differential-drive rover in a walled arena with round obstacles. Used to train the
# reflex net (sim/train.py); the full robot runs in MuJoCo (sim/mj_world.py).
import math
import random

ARENA_W, ARENA_H = 8.0, 6.0      # metres
ROBOT_R = 0.15
WHEEL_BASE = 0.3
MAX_SPEED = 0.6                  # m/s per wheel
SENSOR_ANGLES = (-60, -30, 0, 30, 60)  # degrees, relative to heading (+ = left)
SENSOR_RANGE = 1.5
DT = 0.1
CELL = 0.5                       # coverage grid size


class World:
    def __init__(self, seed=None, n_obstacles=8):
        self.rng = random.Random(seed)
        self.obstacles = []
        while len(self.obstacles) < n_obstacles:
            r = self.rng.uniform(0.2, 0.5)
            x = self.rng.uniform(r + 0.5, ARENA_W - r - 0.5)
            y = self.rng.uniform(r + 0.5, ARENA_H - r - 0.5)
            if all(math.hypot(x - ox, y - oy) > r + orr + 0.6 for ox, oy, orr in self.obstacles):
                self.obstacles.append((x, y, r))
        self.reset_robot()

    def reset_robot(self):
        while True:
            x = self.rng.uniform(0.5, ARENA_W - 0.5)
            y = self.rng.uniform(0.5, ARENA_H - 0.5)
            if all(math.hypot(x - ox, y - oy) > r + ROBOT_R + 0.4 for ox, oy, r in self.obstacles):
                break
        self.x, self.y = x, y
        self.heading = self.rng.uniform(-math.pi, math.pi)
        self.collisions = 0
        self.distance = 0.0
        self.visited = set()

    def _ray(self, angle):
        dx, dy = math.cos(angle), math.sin(angle)
        best = SENSOR_RANGE
        if dx > 1e-9:
            best = min(best, (ARENA_W - self.x) / dx)
        elif dx < -1e-9:
            best = min(best, -self.x / dx)
        if dy > 1e-9:
            best = min(best, (ARENA_H - self.y) / dy)
        elif dy < -1e-9:
            best = min(best, -self.y / dy)
        for ox, oy, r in self.obstacles:
            fx, fy = self.x - ox, self.y - oy
            b = fx * dx + fy * dy
            disc = b * b - (fx * fx + fy * fy - r * r)
            if disc >= 0:
                t = -b - math.sqrt(disc)
                if 0 <= t < best:
                    best = t
        return max(0.0, best - ROBOT_R)

    def sensors(self):
        """Distances as fractions of SENSOR_RANGE (1 = clear)."""
        return [min(1.0, self._ray(self.heading + math.radians(a)) / (SENSOR_RANGE - ROBOT_R))
                for a in SENSOR_ANGLES]

    def step(self, left, right):
        """Wheel commands in -1..1. Returns True if the robot bumped something."""
        vl, vr = max(-1.0, min(1.0, left)) * MAX_SPEED, max(-1.0, min(1.0, right)) * MAX_SPEED
        v = (vl + vr) / 2
        self.heading += (vr - vl) / WHEEL_BASE * DT
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        nx = self.x + v * math.cos(self.heading) * DT
        ny = self.y + v * math.sin(self.heading) * DT
        bumped = not (ROBOT_R <= nx <= ARENA_W - ROBOT_R and ROBOT_R <= ny <= ARENA_H - ROBOT_R)
        if not bumped:
            bumped = any(math.hypot(nx - ox, ny - oy) < r + ROBOT_R for ox, oy, r in self.obstacles)
        if bumped:
            self.collisions += 1
        else:
            self.distance += abs(v) * DT
            self.x, self.y = nx, ny
        self.visited.add((int(self.x / CELL), int(self.y / CELL)))
        return bumped
