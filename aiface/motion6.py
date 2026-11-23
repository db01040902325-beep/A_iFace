"""Six-joint *simulated* trajectory planning, not motor control.

This 6R controller applies limits independently to joint angles; it does
not certify tool clearance, synchronized Cartesian motion, joint torques, or
hardware safety. Emergency stop is an idealized software freeze only.
"""
import csv
import math
from pathlib import Path

from .arm6 import Arm6Config, forward6
from .core import MockMotion, clamp
from .ik6 import inverse6


class MockMotion6:
    """Six independent joint trajectories with per-sample speed/acceleration caps."""

    def __init__(self, cfg=None, initial=(0.,)*6):
        self.cfg = cfg if cfg is not None else Arm6Config()
        if not self.cfg.within_limits(initial):
            raise ValueError('Initial 6R joint angles outside limits')
        self.q = list(initial)
        self.dq = [0.]*6
        self.halted = False

    def emergency_stop(self):
        """Virtual freeze; real motors still require a separate safety circuit."""
        self.halted = True
        self.dq = [0.]*6

    def reset(self):
        """Explicit simulated reset; do not bind to automatic face reacquisition."""
        self.halted = False

    def step(self, target, dt):
        if self.halted:
            return tuple(self.q)
        try:
            target = tuple(float(x) for x in target)
            dt = float(dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Six finite target angles and dt required') from exc
        if not math.isfinite(dt) or not (0. < dt <= .1):
            raise ValueError('Invalid simulation control period')
        if not self.cfg.within_limits(target):
            raise ValueError('6R target outside provisional joint limits')

        vmax = math.radians(self.cfg.max_speed_deg_s)
        dvmax = math.radians(self.cfg.max_accel_deg_s2) * dt
        proposed_q = list(self.q)
        proposed_dq = list(self.dq)
        for j in range(6):
            error = target[j] - self.q[j]
            velocity = self.dq[j]
            if error == 0. and velocity == 0.:
                continue
            direction = (
                1. if error > 0. else -1. if error < 0.
                else (-1. if velocity > 0. else 1.)
            )
            toward = direction * velocity
            low = max(-vmax, toward - dvmax)
            high = min(vmax, toward + dvmax)
            if high <= 0.:
                next_toward = high
            else:
                # Match the tested 3R discrete stopping-distance formulation.
                lo_safe, hi_safe = 0., high
                for _ in range(36):
                    cand = (lo_safe + hi_safe) / 2.
                    stop = cand * dt + MockMotion._braking_distance(
                        cand, dvmax, dt
                    )
                    if stop <= abs(error):
                        lo_safe = cand
                    else:
                        hi_safe = cand
                next_toward = clamp(lo_safe, low, high)
            new_velocity = direction * next_toward
            proposed_q[j] += new_velocity * dt
            proposed_dq[j] = new_velocity

        if not self.cfg.within_limits(proposed_q):
            # Never write an invalid sample or teleport to a boundary.
            self.emergency_stop()
            raise ValueError('6R joint limit fault: halted without moving')
        self.q = proposed_q
        self.dq = proposed_dq
        return tuple(self.q)


def demo(output=Path('results/motion6_demo.csv')):
    """Deterministic 6R IK targets and retargeting with recorded FK pose."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    motion = MockMotion6()
    references = (
        ('approach', (45., 30., -25., -5., 15., 20.), 240),
        ('retarget', (-30., 15., 20., -5., -10., -15.), 270),
        ('home', (0., 0., 0., 0., 0., 0.), 300),
    )
    fieldnames = (
        ['t_s', 'phase', 'x_m', 'y_m', 'z_m']
        + [f'q{j}_deg' for j in range(1, 7)]
        + [f'target_q{j}_deg' for j in range(1, 7)]
    )
    samples = 0
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for phase, target_angles_deg, frames in references:
            reference = tuple(math.radians(v) for v in target_angles_deg)
            pose = forward6(reference, motion.cfg)
            target = inverse6(
                pose.position_m, pose.rotation_matrix,
                cfg=motion.cfg, current=tuple(motion.q)
            )
            for _ in range(frames):
                q = motion.step(target, .02)
                actual = forward6(q, motion.cfg)
                samples += 1
                writer.writerow([
                    round(samples*.02, 5), phase, *actual.position_m,
                    *(math.degrees(v) for v in q),
                    *(math.degrees(v) for v in target),
                ])
    maximum_error = max(abs(q) for q in motion.q)
    print(f'Six-axis simulation samples: {samples}')
    print('Final joint angles (deg):', tuple(round(math.degrees(v), 5) for v in motion.q))
    print(f'Final home joint error: {math.degrees(maximum_error):.6f} deg')
    print('CSV:', output)
    print('SIMULATION ONLY: MOTORS OFF; NO COLLISION OR TORQUE VERIFICATION')
    if maximum_error > math.radians(.001):
        raise RuntimeError('6R simulation did not settle at home')


if __name__ == '__main__':
    demo()
