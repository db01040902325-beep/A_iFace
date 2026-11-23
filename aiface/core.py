"""3R planar robot arm: kinematics, Lagrangian inverse dynamics and safe mock motion.

Coordinates: x forward from shoulder; z upward; all joint angles in radians.
Only a simulated arm. Never use these functions directly as a human-facing motor driver.
"""
from dataclasses import dataclass
from math import atan2, cos, sin, acos, pi, isfinite, ceil


def _finite(*xs):
    if not all(isfinite(float(x)) for x in xs):
        raise ValueError('Non-finite value')


def clamp(x, low, high):
    return max(low, min(high, x))


@dataclass(frozen=True)
class ArmConfig:
    lengths: tuple[float, float, float] = (0.15, 0.13, 0.04)
    masses: tuple[float, float, float] = (0.10, 0.08, 0.04)
    phone_mass: float = 0.20
    phone_size_m: tuple[float, float] = (0.075, 0.155)
    limits_deg: tuple[tuple[float, float], ...] = ((-85, 85), (-145, 145), (-145, 145))
    max_speed_deg_s: float = 20.0
    max_accel_deg_s2: float = 80.0

    def __post_init__(self):
        if len(self.lengths) != 3 or len(self.masses) != 3 or len(self.limits_deg) != 3:
            raise ValueError('Exactly three links are required')
        if any(not isfinite(v) or v <= 0 for v in self.lengths + self.masses + self.phone_size_m):
            raise ValueError('Link parameters must be positive')
        if self.phone_mass <= 0 or self.max_speed_deg_s <= 0 or self.max_accel_deg_s2 <= 0:
            raise ValueError('Motor/phone parameters must be positive')
        if any(not (isfinite(lo) and isfinite(hi) and lo < hi) for lo, hi in self.limits_deg):
            raise ValueError('Invalid joint limits')

    def within_limits(self, q):
        return len(q) == 3 and all(isfinite(v) and lo-1e-9 <= v*180/pi <= hi+1e-9
                                    for v, (lo, hi) in zip(q, self.limits_deg))


def forward(q, cfg=ArmConfig()):
    if len(q) != 3: raise ValueError('Need 3 joint angles')
    _finite(*q)
    a1, a2, a3 = q[0], q[0]+q[1], sum(q)
    l1, l2, l3 = cfg.lengths
    return (l1*cos(a1)+l2*cos(a2)+l3*cos(a3),
            l1*sin(a1)+l2*sin(a2)+l3*sin(a3), a3)


def inverse(x, z, phi, cfg=ArmConfig(), current=(0., 0., 0.)):
    _finite(x, z, phi, *current)
    l1, l2, l3 = cfg.lengths
    wx, wz = x - l3*cos(phi), z - l3*sin(phi)
    c2 = (wx*wx + wz*wz - l1*l1-l2*l2) / (2*l1*l2)
    if not -1-1e-9 <= c2 <= 1+1e-9:
        raise ValueError('Unreachable target')
    a = acos(clamp(c2, -1, 1))
    candidates=[]
    for q2 in (a,-a):
        q1 = atan2(wz, wx) - atan2(l2*sin(q2), l1+l2*cos(q2))
        q3 = phi-q1-q2
        q=(q1,q2,q3)
        if cfg.within_limits(q):
            candidates.append(q)
    if not candidates: raise ValueError('Target violates joint limits')
    return min(candidates, key=lambda q: sum((q[i]-current[i])**2 for i in range(3)))


def _body_jacobians(q, cfg):
    """List (mass, Jx, Jz, rotational inertia, Jomega) for 3 rods + phone."""
    a=[q[0],q[0]+q[1],sum(q)]
    out=[]
    for k in range(3):
        # Uniform slender rod center of mass k
        seg=[cfg.lengths[i] if i<k else cfg.lengths[i]/2 if i==k else 0. for i in range(3)]
        jx=[-sum(seg[i]*sin(a[i]) for i in range(j,k+1)) if j<=k else 0. for j in range(3)]
        jz=[sum(seg[i]*cos(a[i]) for i in range(j,k+1)) if j<=k else 0. for j in range(3)]
        jw=[1. if j<=k else 0. for j in range(3)]
        out.append((cfg.masses[k], jx,jz,cfg.masses[k]*cfg.lengths[k]**2/12,jw))
    # Phone center is assumed at end effector, orientation follows wrist.
    jx=[-sum(cfg.lengths[i]*sin(a[i]) for i in range(j,3)) for j in range(3)]
    jz=[sum(cfg.lengths[i]*cos(a[i]) for i in range(j,3)) for j in range(3)]
    inertia = cfg.phone_mass * sum(side**2 for side in cfg.phone_size_m)/12
    out.append((cfg.phone_mass,jx,jz,inertia,[1.,1.,1.]))
    return out


def mass_matrix(q, cfg=ArmConfig()):
    _finite(*q)
    m=[[0.]*3 for _ in range(3)]
    for mass,jx,jz,inertia,jw in _body_jacobians(q,cfg):
        for i in range(3):
            for j in range(3):
                m[i][j] += mass*(jx[i]*jx[j]+jz[i]*jz[j])+inertia*jw[i]*jw[j]
    return m


def gravity_torque(q, cfg=ArmConfig(), g=9.80665):
    _finite(*q,g)
    return tuple(sum(mass*g*jz[i] for mass,_,jz,_,_ in _body_jacobians(q,cfg)) for i in range(3))


def required_torque(q, dq=(0.,0.,0.), ddq=(0.,0.,0.), cfg=ArmConfig()):
    """Lagrange inverse dynamics M(q)ddq + C(q,dq)dq + G(q).

    Finite-difference Christoffel terms; excludes motor friction, gear losses,
    backlash and external contact. For design calculations only.
    """
    if any(len(v)!=3 for v in (q,dq,ddq)): raise ValueError('3 joints required')
    _finite(*q,*dq,*ddq)
    mat=mass_matrix(q,cfg)
    eps=1e-5
    d=[]
    for k in range(3):
        p=list(q); n=list(q);p[k]+=eps;n[k]-=eps
        mp=mass_matrix(p,cfg); mn=mass_matrix(n,cfg)
        d.append([[(mp[i][j]-mn[i][j])/(2*eps) for j in range(3)] for i in range(3)])
    gravity=gravity_torque(q,cfg)
    return tuple(sum(mat[i][j]*ddq[j] for j in range(3)) + gravity[i] +
                 0.5*sum((d[k][i][j]+d[j][i][k]-d[i][j][k])*dq[j]*dq[k]
                         for j in range(3) for k in range(3)) for i in range(3))


class MockMotion:
    """Software-only velocity and acceleration limited trajectory planner."""
    def __init__(self, cfg=ArmConfig(), initial=(0.,0.,0.)):
        if not cfg.within_limits(initial): raise ValueError('Initial pose outside limits')
        self.cfg=cfg
        self.q=list(initial)
        self.dq=[0.]*3
        self.halted=False

    def emergency_stop(self):
        self.halted=True
        self.dq=[0.]*3

    def reset(self):
        # Does not implement any physical emergency stop reset.
        self.halted=False

    @staticmethod
    def _braking_distance(speed, speed_step, dt):
        """Distance travelled from this speed to rest, with max discrete braking.

        Integrates positions with the same post-update velocity rule as step().
        There is no instantaneous position snap or velocity reset.
        """
        if speed <= 0.:
            return 0.
        n = max(0, ceil(speed / speed_step) - 1)
        return dt * max(0., n * speed - speed_step * n * (n + 1) / 2.)

    def step(self, target, dt):
        """Advance mock joints with hard velocity and *discrete* acceleration caps.

        For a stationary target, use a stopping-distance feasibility check to
        avoid overshooting the target or teleporting the joint at arrival.
        Sudden changes of target may require a normal deceleration/reversal.
        Emergency stop remains a *software-only*, idealized halt.
        """
        if self.halted:
            return tuple(self.q)
        _finite(dt, *target)
        if not 0 < dt <= .1:
            raise ValueError('Invalid control period')
        if not self.cfg.within_limits(target):
            raise ValueError('Joint limit exceeded')

        vmax = self.cfg.max_speed_deg_s * pi / 180.
        dvmax = self.cfg.max_accel_deg_s2 * pi / 180. * dt

        for j in range(3):
            error = target[j] - self.q[j]
            velocity = self.dq[j]
            if error == 0. and velocity == 0.:
                continue

            # Express all quantities in the direction of the current target.
            direction = 1. if error > 0. else -1. if error < 0. else (-1. if velocity > 0. else 1.)
            toward_velocity = direction * velocity

            low = max(-vmax, toward_velocity - dvmax)
            high = min(vmax, toward_velocity + dvmax)
            if high <= 0.:
                next_toward = high  # Still moving away; decelerate or reverse.
            else:
                # Largest next velocity which still allows braking to rest
                # without crossing this stationary target. Binary search
                # resolves exact discrete-time braking (unlike sqrt(2*a*d)).
                lo_safe, hi_safe = 0., high
                distance = abs(error)
                for _ in range(36):
                    candidate = (lo_safe + hi_safe) / 2.
                    stopping = candidate * dt + self._braking_distance(
                        candidate, dvmax, dt
                    )
                    if stopping <= distance:
                        lo_safe = candidate
                    else:
                        hi_safe = candidate

                # A moving, newly retargeted joint may already be unable to
                # brake within the new distance. Never exceed dvmax to hide it.
                next_toward = clamp(lo_safe, low, high)

            next_velocity = direction * next_toward
            self.q[j] += next_velocity * dt
            self.dq[j] = next_velocity

        if not self.cfg.within_limits(self.q):
            self.emergency_stop()
            raise ValueError('Internal joint limit fault')
        return tuple(self.q)
