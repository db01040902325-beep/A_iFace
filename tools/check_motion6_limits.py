"""Check six-joint mock motion CSV by finite differences of positions.

Nominal trajectories only; E-stop deliberately freezes virtual motion
instantaneously, and must be assessed separately rather than with this
normal-operation acceleration check.
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiface.arm6 import Arm6Config


def check(path='results/motion6_demo.csv'):
    cfg = Arm6Config()
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'t_s'} | {f'q{j}_deg' for j in range(1, 7)}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing six joint positions/time columns')
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError('Need at least two motion samples')

    previous_t = 0.
    previous_q = [0.]*6
    previous_v = [0.]*6
    peak_v = [0.]*6
    peak_a = [0.]*6
    for row in rows:
        t = float(row['t_s'])
        dt = t-previous_t
        if not math.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('Non-increasing or invalid sample time')
        angles_deg = [float(row[f'q{j}_deg']) for j in range(1,7)]
        q = [math.radians(v) for v in angles_deg]
        if not cfg.within_limits(q):
            raise ValueError(f'Invalid joint position at time {t}')
        v = [(q[j]-previous_q[j])/dt for j in range(6)]
        a = [(v[j]-previous_v[j])/dt for j in range(6)]
        for j in range(6):
            peak_v[j] = max(peak_v[j], abs(math.degrees(v[j])))
            peak_a[j] = max(peak_a[j], abs(math.degrees(a[j])))
        previous_t, previous_q, previous_v = t, q, v

    passed_all = True
    for j in range(6):
        passed = (
            peak_v[j] <= cfg.max_speed_deg_s + 1e-5
            and peak_a[j] <= cfg.max_accel_deg_s2 + 1e-5
        )
        print(f'Joint {j+1}: speed={peak_v[j]:.3f} deg/s, '
              f'accel={peak_a[j]:.3f} deg/s^2: {"PASS" if passed else "FAIL"}')
        passed_all &= passed
    if not passed_all:
        raise SystemExit(1)


if __name__ == '__main__':
    check(sys.argv[1] if len(sys.argv) > 1 else 'results/motion6_demo.csv')
