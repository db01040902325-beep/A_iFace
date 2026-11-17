"""3D head pose and *approximate* monocular depth -> simulated 3R arm target.

Camera assumption for this research demo: FIXED camera at the arm base, looking
along +X, image top is +Z. This is NOT valid for a camera on a moving phone.
No physical motor control or real collision safety is provided here.
"""
from dataclasses import dataclass
from math import asin, atan2, degrees, hypot, isfinite, pi, radians, sqrt

from .core import ArmConfig, inverse


@dataclass(frozen=True)
class HeadObservation:
    cx: float
    cy: float
    eye_spacing_px: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


def angles_from_matrix(matrix):
    """Rotation matrix -> camera-axis pitch(X), yaw(Y), roll(Z), degrees.

    Euler Z-Y-X convention: R = Rz(roll)*Ry(yaw)*Rx(pitch).
    Coordinates follow the MediaPipe camera image, so a camera-specific
    sign calibration is required before interpreting physical head direction.
    """
    try:
        rows = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError('Expected 3x3/4x4 face rotation matrix') from exc
    if not all(isfinite(x) for row in rows for x in row):
        raise ValueError('Nonfinite face matrix')
    # Remove near-uniform scale; reject reflections/degenerate matrices.
    scales = [sqrt(sum(rows[i][j]**2 for i in range(3))) for j in range(3)]
    if min(scales) < 1e-6 or max(scales) / min(scales) > 1.25:
        raise ValueError('Degenerate face rotation')
    r = [[rows[i][j]/scales[j] for j in range(3)] for i in range(3)]
    det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
           -r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
           +r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if not 0.8 < det < 1.2:
        raise ValueError('Invalid face rotation determinant')
    yaw = atan2(-r[2][0], hypot(r[0][0], r[1][0]))
    pitch = atan2(r[2][1], r[2][2])
    roll = atan2(r[1][0], r[0][0])
    return tuple(degrees(v) for v in (pitch, yaw, roll))


def observation_from_landmarks(landmarks, face_matrix, width, height):
    """MediaPipe 478-point face mesh with iris centers at 468 and 473."""
    if width <= 0 or height <= 0 or len(landmarks) < 478:
        raise ValueError('Need full 478-point face mesh and valid frame')
    a, b = landmarks[468], landmarks[473]
    vals = (a.x, a.y, b.x, b.y)
    if not all(isfinite(float(v)) for v in vals):
        raise ValueError('Invalid iris landmarks')
    cx,cy = (a.x+b.x)/2, (a.y+b.y)/2
    eye_px = hypot((a.x-b.x)*width,(a.y-b.y)*height)
    if not (0 < cx < 1 and 0 < cy < 1 and eye_px >= 4):
        raise ValueError('Eyes outside usable image')
    return HeadObservation(cx, cy, eye_px, *angles_from_matrix(face_matrix))


@dataclass(frozen=True)
class MonoCalibration:
    """Manual reference measurement: person must be at measured distance."""
    reference_distance_m: float
    reference_eye_spacing_px: float
    min_distance_m: float = 0.30
    max_distance_m: float = 1.50

    def __post_init__(self):
        vals = (self.reference_distance_m,self.reference_eye_spacing_px,
                self.min_distance_m,self.max_distance_m)
        if not all(isfinite(x) and x > 0 for x in vals):
            raise ValueError('Positive finite calibration needed')
        if not self.min_distance_m < self.reference_distance_m < self.max_distance_m:
            raise ValueError('Reference distance outside permitted range')

    def estimate(self, observed_eye_px, yaw_deg):
        if not (isfinite(observed_eye_px) and observed_eye_px > 4 and isfinite(yaw_deg)):
            raise ValueError('Invalid eye measurement')
        if abs(yaw_deg) > 22:
            raise ValueError('Turned head: monocular distance unreliable')
        d = self.reference_distance_m * self.reference_eye_spacing_px / observed_eye_px
        if not self.min_distance_m <= d <= self.max_distance_m:
            raise ValueError('Estimated depth out of bounds')
        return d


@dataclass(frozen=True)
class FixedCameraRig:
    width_px: int = 640
    height_px: int = 480
    focal_px: float = 554.3  # EXAMPLE ONLY: approx 60-degree HFOV at 640px
    x_m: float = 0.0
    z_m: float = 0.12

    def __post_init__(self):
        if not (self.width_px > 0 and self.height_px > 0 and
                isfinite(self.focal_px) and self.focal_px > 0 and
                isfinite(self.x_m) and isfinite(self.z_m)):
            raise ValueError('Invalid fixed camera calibration')


@dataclass(frozen=True)
class FollowPlan:
    status: str
    distance_m: float | None = None
    face_point_xz: tuple[float,float] | None = None
    target_pose: tuple[float,float,float] | None = None
    joint_angles: tuple[float,float,float] | None = None


class SimulatedFollower:
    """Plan only; never send joint targets to motors."""
    def __init__(self, calibration: MonoCalibration, rig: FixedCameraRig,
                 cfg: ArmConfig | None = None, desired_standoff_m: float = 0.45):
        if not 0.35 <= desired_standoff_m <= 0.65:
            raise ValueError('Standoff outside prototype configuration')
        self.calibration = calibration
        self.rig = rig
        self.cfg = cfg or ArmConfig()
        self.standoff = desired_standoff_m

    def plan(self, obs: HeadObservation | None, current=(0.,0.,0.)):
        if obs is None:
            return FollowPlan('HOLD_NO_FACE')
        if any(not isfinite(v) for v in (obs.cx,obs.cy,obs.pitch_deg,obs.yaw_deg,obs.roll_deg)):
            return FollowPlan('HOLD_INVALID')
        if not (0 < obs.cx < 1 and 0 < obs.cy < 1):
            return FollowPlan('HOLD_INVALID')
        if abs(obs.pitch_deg) > 25 or abs(obs.yaw_deg) > 22 or abs(obs.roll_deg) > 25:
            return FollowPlan('HOLD_UNSUPPORTED_HEAD_POSE')
        try:
            d = self.calibration.estimate(obs.eye_spacing_px,obs.yaw_deg)
        except ValueError:
            return FollowPlan('HOLD_UNRELIABLE_DEPTH')
        # CAMERA MUST BE FIXED TO ARM BASE; moving camera requires dynamic FK extrinsics.
        lateral = (obs.cx - .5)*self.rig.width_px/self.rig.focal_px*d
        if abs(lateral) > 0.04:
            return FollowPlan('HOLD_OUTSIDE_2D_PLANE',d)
        fx = self.rig.x_m + d
        fz = self.rig.z_m + (.5-obs.cy)*self.rig.height_px/self.rig.focal_px*d
        goal_x = fx-self.standoff
        goal_z = fz
        # Pitch sign is an illustrative choice; must calibrate to camera orientation.
        phi = radians(-obs.pitch_deg)
        # Geometric face-phone standoff; physical collisions still require independent sensors.
        if goal_x < 0.06 or goal_z < -0.05 or hypot(goal_x-fx,goal_z-fz) < 0.35:
            return FollowPlan('HOLD_SAFETY_ENVELOPE',d,(fx,fz))
        try:
            angles = inverse(goal_x,goal_z,phi,self.cfg,current=current)
        except ValueError:
            return FollowPlan('HOLD_IK_UNREACHABLE',d,(fx,fz))
        return FollowPlan('TRACK_SIMULATION',d,(fx,fz),(goal_x,goal_z,phi),angles)
