"""6R face-to-joint simulation: camera must be fixed on robot base.

This is an *illustrative* mapping, not a calibrated camera-to-robot transform.
World axes: +X forward, +Y horizontal camera-right (assumed), +Z up.
Head pitch/yaw/roll signs and phone mount orientation MUST be calibrated.
Distance from eye spacing is approximate monocular depth, not a depth sensor.
NO physical motor control, contact safety, or collision avoidance here.
"""
from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite, pi, radians, sin
from pathlib import Path
import csv

from .arm6 import Arm6Config, forward6
from .head_pose import HeadObservation, MonoCalibration, FixedCameraRig
from .ik6 import inverse6
from .motion6 import MockMotion6


@dataclass(frozen=True)
class FacePlan6:
    status: str
    distance_m: float | None = None
    face_xyz_m: tuple[float, float, float] | None = None
    target_xyz_m: tuple[float, float, float] | None = None
    target_joint_angles: tuple[float, ...] | None = None


class FaceFollower6:
    """Compute tentative 6R targets; never sends commands to motors."""

    def __init__(self, calibration, rig, cfg=None, desired_standoff_m=.45):
        if calibration is not None and not isinstance(calibration, MonoCalibration):
            raise ValueError('Use a valid mono-camera calibration')
        if not isinstance(rig, FixedCameraRig):
            raise ValueError('Camera rig required')
        if not isfinite(desired_standoff_m) or not .35 <= desired_standoff_m <= .65:
            raise ValueError('Standoff outside 35-65cm preview range')
        self.calibration = calibration
        self.rig = rig
        self.cfg = cfg or Arm6Config()
        self.standoff = desired_standoff_m

    def plan(self, observation: HeadObservation | None, current=(0.,)*6, *, faces=1):
        if faces == 0:
            return FacePlan6('HOLD_NO_FACE')
        if faces != 1:
            return FacePlan6('HOLD_MULTIPLE_FACES')
        if observation is None:
            return FacePlan6('HOLD_NO_FACE_MATRIX')
        o = observation
        if any(not isfinite(x) for x in (
            o.cx, o.cy, o.eye_spacing_px, o.pitch_deg, o.yaw_deg, o.roll_deg
        )) or not (0 < o.cx < 1 and 0 < o.cy < 1):
            return FacePlan6('HOLD_INVALID_OBSERVATION')
        if (abs(o.pitch_deg)>25 or abs(o.yaw_deg)>22 or abs(o.roll_deg)>25):
            return FacePlan6('HOLD_UNSUPPORTED_HEAD_POSE')
        if self.calibration is None:
            return FacePlan6('HOLD_NEEDS_CALIBRATION')
        try:
            distance = self.calibration.estimate(o.eye_spacing_px,o.yaw_deg)
        except ValueError:
            return FacePlan6('HOLD_UNRELIABLE_DEPTH')

        # Approximate optical projection; assumes square pixels and zero distortion.
        y = (o.cx-.5)*self.rig.width_px/self.rig.focal_px*distance
        z = self.rig.z_m + (.5-o.cy)*self.rig.height_px/self.rig.focal_px*distance
        x = self.rig.x_m+distance
        face = (x, y, z)
        horizontal_range = hypot(distance,y)
        if horizontal_range <= self.standoff:
            return FacePlan6('HOLD_SAFETY_ENVELOPE',distance,face)
        ratio = (horizontal_range-self.standoff)/horizontal_range
        goal = (self.rig.x_m+distance*ratio, y*ratio, z)
        if goal[0] < .06 or goal[2] < -.05 or goal[2] > .46:
            return FacePlan6('HOLD_SAFETY_ENVELOPE',distance,face,goal)

        # IMPORTANT: design-only head pose mapping.
        # q0 aims the arm into the face azimuth; q3 sets the screen pitch.
        # Wrist local X (roll) and Z (yaw) are NOT guaranteed to match
        # perceived phone screen rotations on any uncalibrated real mounting.
        yaw_arm = atan2(y, distance)
        preview_angles = (
            yaw_arm, 0., 0., radians(-o.pitch_deg),
            radians(o.roll_deg), radians(o.yaw_deg)
        )
        try:
            desired_rotation = forward6(preview_angles,self.cfg).rotation_matrix
            q = inverse6(goal, desired_rotation, self.cfg,current=current)
        except ValueError:
            return FacePlan6('HOLD_IK_UNREACHABLE',distance,face,goal)
        return FacePlan6('TRACK_SIMULATION_6R',distance,face,goal,q)


class SimulatedFace6Controller:
    """Fail-closed 6R preview with manual stop latch and reacquire delay.

    A loss/ambiguity of face forces a virtual HOLD at exact current q.
    The manual stop is latched until the user explicitly calls reset_manual().
    After visual hold, five *consecutive* valid plans are needed to resume.
    """

    def __init__(self, follower, motion=None, reacquire_frames=5):
        if not isinstance(follower, FaceFollower6):
            raise ValueError('6R follower required')
        if not isinstance(reacquire_frames,int) or reacquire_frames<1:
            raise ValueError('At least one valid reacquire frame required')
        self.follower=follower
        self.motion=motion or MockMotion6(follower.cfg)
        self.reacquire_frames=reacquire_frames
        self.valid_streak=0
        self.manual_latched=False

    def manual_stop(self):
        self.manual_latched=True
        self.valid_streak=0
        self.motion.emergency_stop()

    def reset_manual(self):
        # Clears only the manual latch; visual validity must be rechecked.
        self.manual_latched=False
        self.valid_streak=0

    def update(self, observation, *, faces=1, dt=.02):
        # Invalid control period must not inadvertently resume motion.
        if not isfinite(dt) or not .0 < dt <= .1:
            self.motion.emergency_stop()
            self.valid_streak=0
            return FacePlan6('HOLD_INVALID_DT')
        plan=self.follower.plan(observation,tuple(self.motion.q),faces=faces)
        if self.manual_latched:
            self.motion.emergency_stop()
            return FacePlan6('HOLD_MANUAL_STOP',plan.distance_m,
                             plan.face_xyz_m,plan.target_xyz_m)
        if plan.status != 'TRACK_SIMULATION_6R':
            self.valid_streak=0
            self.motion.emergency_stop()
            return plan
        self.valid_streak += 1
        if self.valid_streak < self.reacquire_frames:
            self.motion.emergency_stop()
            return FacePlan6('HOLD_REACQUIRING',plan.distance_m,
                             plan.face_xyz_m,plan.target_xyz_m)
        try:
            if self.motion.halted:
                self.motion.reset()
            self.motion.step(plan.target_joint_angles,dt)
            return plan
        except ValueError:
            self.motion.emergency_stop()
            self.valid_streak=0
            return FacePlan6('HOLD_MOTION_FAULT',plan.distance_m,
                             plan.face_xyz_m,plan.target_xyz_m)


def synthetic_demo(output=Path('results/follow6_demo.csv')):
    """Reproducible, network-free face-movement and 6R safety-hold test."""
    from collections import Counter
    from math import cos
    cfg=Arm6Config()
    follower=FaceFollower6(MonoCalibration(.65,64),FixedCameraRig(),cfg)
    ctrl=SimulatedFace6Controller(follower)
    counts=Counter()
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['time_s','status','faces','estimated_distance_m',
                         'face_x_m','face_y_m','face_z_m',
                         'x_m','y_m','z_m',
                         *[f'q{j}_deg' for j in range(1,7)]])
        for i in range(400):
            t=i*.02
            dist=.65+.03*sin(t*1.1)
            cx=.5+.09*sin(t*.8)
            cy=.5+.025*cos(t*.65)
            obs=HeadObservation(
                cx,cy,64*.65/dist,
                3*sin(t*.3), 4*sin(t*.4), 2*sin(t*.7)
            )
            faces=1
            if 90 <= i < 110:
                faces,obs=0,None
            elif 230 <= i < 240:
                faces=2
            plan=ctrl.update(obs,faces=faces,dt=.02)
            counts[plan.status]+=1
            pose=forward6(ctrl.motion.q,cfg)
            face=plan.face_xyz_m or ('','','')
            writer.writerow([
                round((i+1)*.02,3),plan.status,faces,
                plan.distance_m if plan.distance_m is not None else '',
                *face,*pose.position_m,
                *(degrees(a) for a in ctrl.motion.q)
            ])
    print('6R FACE FOLLOW (SYNTHETIC):',dict(counts))
    print('CSV:',output)
    print('SIMULATION ONLY: NO WEBCAM / NO PHYSICAL MOTORS')
    if counts['TRACK_SIMULATION_6R'] < 50 or counts['HOLD_NO_FACE'] != 20 or counts['HOLD_MULTIPLE_FACES'] != 10:
        raise RuntimeError('6R synthetic face-follow state regression')


if __name__ == '__main__':
    synthetic_demo()
