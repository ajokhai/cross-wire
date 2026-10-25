# MuJoCo world for any wheeled URDF robot: loads the URDF, finds its wheels from the
# joint layout, adds velocity motors, five range sensors, odometry and a camera that
# reports coloured objects in view, and builds an arena. The ESP32 brain only ever sees
# ranges, pose, detections and sends wheel commands.
import math
import os
import random
import re

import mujoco
import numpy as np

SENSOR_ANGLES = (-60, -30, 0, 30, 60)   # degrees, + = left
SENSOR_RANGE = 1.5                      # metres; readings are sent as 0..100 of this
CAMERA_FOV = 60                         # degrees either side of straight ahead
CAMERA_RANGE = 6.0                      # metres
ARENA = 10.0                            # square arena, metres
PHYSICS_DT = 0.002
CONTROL_DT = 0.1                        # the brain is asked 10 times a second

COLORS = {"red": (0.85, 0.2, 0.2), "blue": (0.2, 0.4, 0.95), "green": (0.2, 0.7, 0.3),
          "yellow": (0.95, 0.85, 0.2), "orange": (0.95, 0.55, 0.15), "purple": (0.6, 0.3, 0.8)}


def load_urdf(path):
    """URDF -> MjSpec. Expands leftover xacro defaults like $(optenv NAME 0 0 1)."""
    folder = os.path.dirname(os.path.abspath(path))
    xml = open(path).read()
    xml = re.sub(r"\$\(optenv \S+ ([^)]*)\)", r"\1", xml)
    assets = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".stl", ".obj", ".msh", ".png")):
                assets[f] = open(os.path.join(root, f), "rb").read()
    return mujoco.MjSpec.from_string(xml, assets=assets)


class RobotWorld:
    def __init__(self, urdf, seed=7, n_objects=10, max_speed=1.0):
        self.rng = random.Random(seed)
        self.max_speed = max_speed
        robot = load_urdf(urdf)
        spec = mujoco.MjSpec()
        spec.option.timestep = PHYSICS_DT
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self._scene(spec, n_objects)

        # Attach the robot on a free joint so it can drive around.
        base = robot.worldbody.first_body()
        frame = spec.worldbody.add_frame(pos=[0, 0, 0.05])
        self.root = frame.attach_body(base, "robot/", "")
        self.root.add_freejoint(name="robot/free")

        # Wheels: hinge joints spinning about the robot's sideways (y) axis.
        self.wheels = []  # (joint name, side +1 left / -1 right)
        for body in self.root.find_all(mujoco.mjtObj.mjOBJ_BODY):
            for j in body.joints:
                if j.type == mujoco.mjtJoint.mjJNT_HINGE and abs(j.axis[1]) > 0.9 and abs(body.pos[1]) > 0.05:
                    self.wheels.append((j.name, 1 if body.pos[1] > 0 else -1))
        if not self.wheels:
            raise ValueError("no wheels found in %s" % urdf)
        self.wheel_radius = 0.165
        for name, _ in self.wheels:
            act = spec.add_actuator(name=name + "/motor", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_velocity(kv=40.0)
            act.forcerange = [-60, 60]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

        # Range sensors fanned out from just above the front bumper (further forward they
        # start inside whatever the robot is touching; lower they hit the robot itself).
        for a in SENSOR_ANGLES:
            r = math.radians(a)
            site = self.root.add_site(name="robot/range%d" % a, pos=[0.45, 0, 0.42],
                                      zaxis=[math.cos(r), math.sin(r), 0], size=[0.01, 0, 0])
            sensor = spec.add_sensor(name="range%d" % a, type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                                     objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name)
            sensor.intprm[0] = 1 << int(mujoco.mjtRayDataField.mjRAYDATA_DIST)  # return distance only

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.motor_ids = [self.model.actuator(n + "/motor").id for n, _ in self.wheels]
        self.sides = [s for _, s in self.wheels]
        self.base_id = self.model.body(self.root.name).id
        for o in self.objects:
            o["geom"] = self.model.geom(o["geom_name"]).id
        mujoco.mj_forward(self.model, self.data)
        self.time = 0.0
        self.trail = []
        self.bumps = 0

    def _scene(self, spec, n_objects):
        spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
        spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.25, 0.27, 0.3], rgb2=[0.3, 0.32, 0.35], width=512, height=512)
        spec.add_material(name="grid", textures=["", "grid"], texrepeat=[10, 10])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[ARENA, ARENA, 0.1], material="grid",
                                friction=[1.0, 0.01, 0.001])
        h = ARENA / 2
        for x, y, sx, sy in ((0, h, h, 0.1), (0, -h, h, 0.1), (h, 0, 0.1, h), (-h, 0, 0.1, h)):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, 0.4], size=[sx, sy, 0.4],
                                    rgba=[0.55, 0.55, 0.6, 1])
        # Coloured objects. The first is always a blue box, so "the blue box" exists.
        self.objects = []
        names = list(COLORS)
        while len(self.objects) < n_objects:
            x, y = self.rng.uniform(-h + 1, h - 1), self.rng.uniform(-h + 1, h - 1)
            r = self.rng.uniform(0.25, 0.5)
            if math.hypot(x, y) < 2.0 or any(math.hypot(x - o["x"], y - o["y"]) < r + o["r"] + 1.2 for o in self.objects):
                continue
            first = not self.objects
            color = "blue" if first else self.rng.choice(names)
            shape = "box" if first else self.rng.choice(("box", "cylinder"))
            i = len(self.objects) + 1
            geom_name = "obj%d" % i
            rgba = list(COLORS[color]) + [1]
            if shape == "cylinder":
                spec.worldbody.add_geom(name=geom_name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, 0.4],
                                        size=[r, 0.4, 0], rgba=rgba)
            else:
                spec.worldbody.add_geom(name=geom_name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, 0.4],
                                        size=[r, r, 0.4], euler=[0, 0, self.rng.uniform(0, 90)], rgba=rgba)
            self.objects.append({"id": i, "color": color, "shape": shape, "x": x, "y": y, "r": r,
                                 "geom_name": geom_name})

    def pose(self):
        x, y = self.data.xpos[self.base_id][:2]
        w, qx, qy, qz = self.data.xquat[self.base_id]
        heading = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return float(x), float(y), heading

    def ranges(self):
        """Fractions of SENSOR_RANGE, 1 = clear (rangefinder returns -1 for no hit)."""
        out = []
        for a in SENSOR_ANGLES:
            d = self.data.sensor("range%d" % a).data[0]
            out.append(1.0 if d < 0 or d > SENSOR_RANGE else float(d) / SENSOR_RANGE)
        return out

    def visible(self):
        """Objects the camera can see now: in the field of view, in range, nothing in the way."""
        x, y, h = self.pose()
        eye = np.array([x + 0.3 * math.cos(h), y + 0.3 * math.sin(h), 0.6])
        geomid = np.array([-1], dtype=np.int32)
        seen = []
        for o in self.objects:
            dx, dy = o["x"] - x, o["y"] - y
            dist = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(dy, dx) - h)
            bearing = (bearing + 180) % 360 - 180
            if dist > CAMERA_RANGE or abs(bearing) > CAMERA_FOV:
                continue
            vec = np.array([o["x"], o["y"], 0.4]) - eye
            mujoco.mj_ray(self.model, self.data, eye, vec, None, 1, self.base_id, geomid)
            if geomid[0] == o["geom"]:
                seen.append(o)
        return seen

    def step(self, left, right):
        """Wheel commands in -1..1 for one control period."""
        for i, side in zip(self.motor_ids, self.sides):
            cmd = left if side > 0 else right
            self.data.ctrl[i] = max(-1, min(1, cmd)) * self.max_speed / self.wheel_radius
        before = self.pose()
        for _ in range(int(CONTROL_DT / PHYSICS_DT)):
            mujoco.mj_step(self.model, self.data)
        self.time += CONTROL_DT
        x, y, _ = self.pose()
        if left * right > 0 and abs(left + right) > 0.3 and math.hypot(x - before[0], y - before[1]) < 0.005:
            self.bumps += 1   # driving (not spinning on the spot) but not moving
        self.trail.append((x, y))
        del self.trail[:-500]
