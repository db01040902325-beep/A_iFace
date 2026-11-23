"""Six-revolute-joint (6R) arm: 3D forward kinematics only.

Assumed chain: base yaw Z, shoulder/elbow/wrist pitch local Y,
tool roll local X, tool yaw local Z. All angles are in radians.
World Z points up, world X points forward at zero base yaw.
The camera, phone mounting, tool offsets, motor masses, collision geometry,
and any physical controller are NOT modeled. No motor control is implemented.
"""
from dataclasses import dataclass
from math import cos, degrees, isfinite, pi, radians, sin


@dataclass(frozen=True)
class Arm6Config:
    lengths_m: tuple[float, float, float] = (0.15, 0.13, 0.04)
    base_height_m: float = 0.12
    # Provisional SOFTWARE limits: replace with actual mechanical stops.
    limits_deg: tuple[tuple[float, float], ...] = (
        (-170., 170.), (-85., 85.), (-145., 145.),
        (-145., 145.), (-160., 160.), (-170., 170.)
    )

    def __post_init__(self):
        if len(self.lengths_m) != 3 or not all(
            isfinite(x) and x > 0 for x in self.lengths_m
        ):
            raise ValueError("Exactly three positive finite link lengths required")
        if not isfinite(self.base_height_m) or self.base_height_m < 0:
            raise ValueError("Invalid base height")
        if len(self.limits_deg) != 6 or any(
            len(pair) != 2 or not all(isfinite(v) for v in pair) or pair[0] >= pair[1]
            for pair in self.limits_deg
        ):
            raise ValueError("Six valid joint ranges required")

    def within_limits(self, angles_rad):
        return (
            len(angles_rad) == 6
            and all(
                isfinite(q) and lo - 1e-9 <= degrees(q) <= hi + 1e-9
                for q, (lo, hi) in zip(angles_rad, self.limits_deg)
            )
        )


@dataclass(frozen=True)
class Arm6Pose:
    position_m: tuple[float, float, float]
    rotation_matrix: tuple[tuple[float, float, float], ...]
    # Fixed physical joint-center assumptions: shoulder, elbow, wrist, tool tip.
    points_m: tuple[tuple[float, float, float], ...]


def _matmul3(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _rz(a):
    c, s = cos(a), sin(a)
    return ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))


def _ry(a):
    c, s = cos(a), sin(a)
    return ((c, 0., s), (0., 1., 0.), (-s, 0., c))


def _rx(a):
    c, s = cos(a), sin(a)
    return ((1., 0., 0.), (0., c, -s), (0., s, c))


def forward6(angles_rad, cfg=Arm6Config()):
    """Return tool center position/orientation for the assumed 6R chain.

    Rotation convention: Rz(q0) Ry(-(q1+q2+q3)) Rx(q4) Rz(q5).
    Positive pitch raises an initially forward horizontal link.
    Joint 5/6 axes intersect at the current tool tip (no offsets yet).
    """
    if not cfg.within_limits(angles_rad):
        raise ValueError("Six finite joint angles within provisional limits required")
    q0, q1, q2, q3, q4, q5 = angles_rad
    yaw_cos, yaw_sin = cos(q0), sin(q0)
    radial, height = 0., cfg.base_height_m
    points = [(0., 0., height)]
    cumulative_pitch = 0.
    for length, angle in zip(cfg.lengths_m, (q1, q2, q3)):
        cumulative_pitch += angle
        radial += length * cos(cumulative_pitch)
        height += length * sin(cumulative_pitch)
        points.append((radial * yaw_cos, radial * yaw_sin, height))
    orient = _matmul3(
        _matmul3(
            _matmul3(_rz(q0), _ry(-cumulative_pitch)), _rx(q4)
        ),
        _rz(q5)
    )
    return Arm6Pose(points[-1], orient, tuple(points))


def _example():
    cfg = Arm6Config()
    for label, deg in (
        ("zero", (0, 0, 0, 0, 0, 0)),
        ("base yaw +90", (90, 0, 0, 0, 0, 0)),
        ("shoulder pitch +30", (0, 30, 0, 0, 0, 0)),
        ("combined", (45, 30, -25, -5, 15, 20)),
    ):
        pose = forward6(tuple(radians(v) for v in deg), cfg)
        print(
            f"{label}: x={pose.position_m[0]:.4f} m "
            f"y={pose.position_m[1]:.4f} m "
            f"z={pose.position_m[2]:.4f} m"
        )
    print("6R FORWARD KINEMATICS ONLY - NO IK, MOTORS, OR COLLISION CHECKS")


if __name__ == "__main__":
    _example()
