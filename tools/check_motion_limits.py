"""Check recorded demo.csv joint velocities and accelerations.

Reconstructs sample velocities from positions as in the reported regression.
Normal emergency-stop states are not included in this standard motion test.
"""
import csv
import math
import sys
from pathlib import Path

from aiface.core import ArmConfig


def check(path):
    cfg=ArmConfig()
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)<2:
        raise ValueError('Expected at least two motion samples')
    previous_t=0.
    previous_q=[0.]*3
    previous_v=[0.]*3
    max_speed=[0.]*3
    max_accel=[0.]*3
    for i, row in enumerate(rows,1):
        t=float(row['t_s'])
        dt=t-previous_t
        if not math.isfinite(dt) or dt<=0 or dt>.1+1e-9:
            raise ValueError(f'Invalid sample time at {i}: {dt}')
        q=[math.radians(float(row[f'q{j}_deg'])) for j in range(1,4)]
        if not cfg.within_limits(q):
            raise ValueError(f'Joint limit violated at {i}')
        v=[(q[j]-previous_q[j])/dt for j in range(3)]
        a=[(v[j]-previous_v[j])/dt for j in range(3)]
        for j in range(3):
            max_speed[j]=max(max_speed[j],abs(math.degrees(v[j])))
            max_accel[j]=max(max_accel[j],abs(math.degrees(a[j])))
        previous_t,previous_q,previous_v=t,q,v
    ok=True
    for j in range(3):
        passed=max_speed[j]<=cfg.max_speed_deg_s+1e-5 and max_accel[j]<=cfg.max_accel_deg_s2+1e-5
        print(f'Joint {j+1}: speed={max_speed[j]:.3f} deg/s, '
              f'accel={max_accel[j]:.3f} deg/s^2: {"PASS" if passed else "FAIL"}')
        ok &= passed
    if not ok:
        sys.exit(1)


if __name__=='__main__':
    check(sys.argv[1] if len(sys.argv)>1 else 'results/demo.csv')
