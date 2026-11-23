"""Inverse kinematics for the provisional six-revolute-joint arm.

Analytic position/orientation decomposition for the exact axis convention in
arm6.forward6. No optimization dependencies or physical motor commands.
Axis and wrist singularities are explicitly rejected rather than choosing an
unpredictable orientation. Collision checking and measured geometry pending.
"""
from math import acos, asin, atan2, cos, degrees, hypot, isfinite, pi, sin

from .arm6 import Arm6Config, forward6, _matmul3, _rz


def _equivalent(angle, limits):
    lo, hi = (v*pi/180 for v in limits)
    return tuple(angle + 2*pi*k for k in range(-2, 3)
                 if lo - 1e-9 <= angle+2*pi*k <= hi + 1e-9)


def _rotation(rotation):
    try:
        R = tuple(tuple(float(v) for v in row) for row in rotation)
        if len(R) != 3 or any(len(row) != 3 for row in R):
            raise ValueError('3x3 rotation matrix required')
    except (TypeError, ValueError) as e:
        raise ValueError('3x3 rotation matrix required') from e
    if not all(isfinite(v) for row in R for v in row):
        raise ValueError('Rotation contains non-finite values')
    for i in range(3):
        for j in range(3):
            dot = sum(R[k][i]*R[k][j] for k in range(3))
            if abs(dot - (1. if i == j else 0.)) > 1e-5:
                raise ValueError('Rotation matrix must be orthonormal')
    determinant = (R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])
                 -R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])
                 +R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
    if abs(determinant - 1.) > 1e-5:
        raise ValueError('Rotation must have determinant +1')
    return R


def inverse6(position_m, rotation_matrix, cfg=Arm6Config(), *, current=None):
    """Return six joint angles (radians) for a target tool pose.

    The input target is expressed in the fixed world frame of forward6.
    Candidate base-yaw, elbow and wrist Euler branches are searched, then
    checked by FK. The feasible solution nearest ``current`` is returned.
    Targets on the base Z-axis and wrist-roll gimbal singularities reject.
    This is a geometry-only IK, not a safety-approved robot controller.
    """
    try:
        position = tuple(float(v) for v in position_m)
    except (TypeError, ValueError) as exc:
        raise ValueError('Expected finite (x,y,z) coordinates') from exc
    if len(position) != 3 or not all(isfinite(v) for v in position):
        raise ValueError('Expected finite (x,y,z) coordinates')
    R = _rotation(rotation_matrix)
    if current is None:
        current = (0.,) * 6
    try:
        current = tuple(float(v) for v in current)
    except (TypeError, ValueError) as exc:
        raise ValueError('Expected six finite current angles') from exc
    if len(current) != 6 or not all(isfinite(v) for v in current):
        raise ValueError('Expected six finite current angles')

    x, y, z = position
    radial = hypot(x, y)
    if radial < 1e-8:
        raise ValueError('Base yaw singularity at x=y=0')
    azimuth = atan2(y, x)
    l1, l2, l3 = cfg.lengths_m
    candidates = []
    for az, signed_radius in ((azimuth, radial),
                              (azimuth + pi, -radial),
                              (azimuth - pi, -radial)):
        for q0 in _equivalent(az, cfg.limits_deg[0]):
            # Rz(-q0) R = Ry(-phi) Rx(roll) Rz(tool_yaw)
            qrel = _matmul3(_rz(-q0), R)
            signed_sin_roll = max(-1., min(1., -qrel[1][2]))
            base_roll = asin(signed_sin_roll)
            if abs(cos(base_roll)) < 1e-7:
                # Euler singularity: infinitely many phi/tool-yaw pairs.
                # Reject rather than using an unstable, arbitrary solution.
                continue
            phi = atan2(-qrel[0][2], qrel[2][2])
            tool_yaw = atan2(qrel[1][0], qrel[1][1])
            other_roll = pi - base_roll if base_roll >= 0. else -pi - base_roll
            branches = ((phi, base_roll, tool_yaw),
                        (phi + pi, other_roll, tool_yaw + pi))
            for phi_base, roll_base, yaw_base in branches:
                for q4 in _equivalent(roll_base, cfg.limits_deg[4]):
                    for q5 in _equivalent(yaw_base, cfg.limits_deg[5]):
                        for k in (-1, 0, 1):
                            phi_total = phi_base + 2*pi*k
                            wx = signed_radius - l3*cos(phi_total)
                            wz = (z - cfg.base_height_m) - l3*sin(phi_total)
                            cos_elbow = ((wx*wx + wz*wz - l1*l1 - l2*l2)
                                         / (2*l1*l2))
                            if abs(cos_elbow) > 1. + 1e-9:
                                continue
                            cos_elbow = max(-1., min(1., cos_elbow))
                            for q2 in (acos(cos_elbow), -acos(cos_elbow)):
                                q1 = (atan2(wz, wx)
                                      - atan2(l2*sin(q2), l1+l2*cos(q2)))
                                q3 = phi_total - q1 - q2
                                angles = (q0, q1, q2, q3, q4, q5)
                                if not cfg.within_limits(angles):
                                    continue
                                solved = forward6(angles, cfg)
                                pos_err = max(abs(a-b) for a,b in
                                              zip(solved.position_m,position))
                                rot_err = max(abs(solved.rotation_matrix[i][j]-R[i][j])
                                              for i in range(3) for j in range(3))
                                if pos_err < 1e-7 and rot_err < 1e-7:
                                    candidates.append(angles)

    if not candidates:
        raise ValueError('No nonsingular IK solution within joint limits')
    return min(candidates, key=lambda q: sum((a-b)**2 for a,b in zip(q,current)))


def _example():
    from math import radians
    actual = tuple(map(radians, (45,30,-25,-5,15,20)))
    target = forward6(actual)
    recovered = inverse6(target.position_m, target.rotation_matrix,current=actual)
    check = forward6(recovered)
    print('target position:', *(f'{v:.5f}' for v in target.position_m))
    print('IK joints (deg):', *(f'{degrees(v):.4f}' for v in recovered))
    print('FK check position:', *(f'{v:.5f}' for v in check.position_m))
    print('6R IK geometry demo only; motors OFF')


if __name__ == '__main__':
    _example()
