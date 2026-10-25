# Tiny policy network: 5 range sensors -> 6 tanh hidden -> 2 wheel speeds.
# Pure Python so the same file runs in CPython (training) and MicroPython (ESP32).
import math

N_IN, N_HID, N_OUT = 5, 6, 2
N_WEIGHTS = (N_IN + 1) * N_HID + (N_HID + 1) * N_OUT  # 50


def act(w, sensors):
    """sensors: 5 floats in 0..1 (1 = nothing in range). Returns (left, right) in -1..1."""
    hidden = []
    k = 0
    for _ in range(N_HID):
        s = w[k]  # bias
        k += 1
        for x in sensors:
            s += w[k] * x
            k += 1
        hidden.append(math.tanh(s))
    out = []
    for _ in range(N_OUT):
        s = w[k]
        k += 1
        for h in hidden:
            s += w[k] * h
            k += 1
        out.append(math.tanh(s))
    return out[0], out[1]
