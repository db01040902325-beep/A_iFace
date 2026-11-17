import math
import unittest
from dataclasses import dataclass
from aiface.head_pose import (
    angles_from_matrix, observation_from_landmarks, MonoCalibration, FixedCameraRig,
    HeadObservation, SimulatedFollower
)
from aiface.pose_camera import _save_calibration, _read_calibration
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass
class Point:
    x: float = .5
    y: float = .5


def sample(distance_eye_px=64.):
    # 640x480, simulated fixed webcam, 65 cm reference.
    return HeadObservation(.5,.5,distance_eye_px,0.,0.,0.)


class PoseAnglesTest(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(tuple(round(x,5) for x in angles_from_matrix(
            ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)))),(0.,0.,0.))

    def test_pitch(self):
        a=math.radians(18)
        r=((1,0,0),(0,math.cos(a),-math.sin(a)),(0,math.sin(a),math.cos(a)))
        self.assertAlmostEqual(angles_from_matrix(r)[0],18)

    def test_yaw(self):
        a=math.radians(-14)
        r=((math.cos(a),0,math.sin(a)),(0,1,0),(-math.sin(a),0,math.cos(a)))
        self.assertAlmostEqual(angles_from_matrix(r)[1],-14)

    def test_roll(self):
        a=math.radians(23)
        r=((math.cos(a),-math.sin(a),0),(math.sin(a),math.cos(a),0),(0,0,1))
        self.assertAlmostEqual(angles_from_matrix(r)[2],23)

    def test_reflection_rejected(self):
        with self.assertRaises(ValueError):
            angles_from_matrix(((1,0,0),(0,1,0),(0,0,-1)))

    def test_bad_matrix_rejected(self):
        with self.assertRaises(ValueError):
            angles_from_matrix(((1,0),(0,1)))

    def test_landmarks(self):
        pts=[Point() for _ in range(478)]
        pts[468]=Point(.45,.5)
        pts[473]=Point(.55,.5)
        o=observation_from_landmarks(pts,((1,0,0),(0,1,0),(0,0,1)),640,480)
        self.assertAlmostEqual(o.eye_spacing_px,64)
        self.assertAlmostEqual(o.cx,.5)
        self.assertAlmostEqual(o.cy,.5)

    def test_missing_iris(self):
        with self.assertRaises(ValueError):
            observation_from_landmarks([Point()]*300,((1,0,0),(0,1,0),(0,0,1)),640,480)


class CalibrationTest(unittest.TestCase):
    def test_reference_depth(self):
        cal=MonoCalibration(.65,64)
        self.assertAlmostEqual(cal.estimate(64,0),.65)

    def test_depth_scales(self):
        cal=MonoCalibration(.65,64)
        self.assertAlmostEqual(cal.estimate(52,0),.8)

    def test_yaw_rejected(self):
        with self.assertRaises(ValueError):
            MonoCalibration(.65,64).estimate(64,30)

    def test_uncalibrated_bounds(self):
        with self.assertRaises(ValueError):
            MonoCalibration(0,64)

    def test_file_roundtrip(self):
        with TemporaryDirectory() as folder:
            f=Path(folder)/'cal.json'
            _save_calibration(f,MonoCalibration(.65,64))
            result=_read_calibration(f)
            self.assertAlmostEqual(result.reference_distance_m,.65)
            self.assertAlmostEqual(result.reference_eye_spacing_px,64)


class FollowTest(unittest.TestCase):
    def setUp(self):
        self.follower=SimulatedFollower(MonoCalibration(.65,64),FixedCameraRig())

    def test_track_center(self):
        plan=self.follower.plan(sample())
        self.assertEqual(plan.status,'TRACK_SIMULATION')
        self.assertAlmostEqual(plan.distance_m,.65)
        self.assertAlmostEqual(plan.target_pose[0],.20)
        self.assertAlmostEqual(plan.target_pose[1],.12)
        self.assertEqual(len(plan.joint_angles),3)

    def test_hold_no_face(self):
        self.assertEqual(self.follower.plan(None).status,'HOLD_NO_FACE')

    def test_hold_yaw(self):
        bad=HeadObservation(.5,.5,64,0,35,0)
        self.assertEqual(self.follower.plan(bad).status,'HOLD_UNSUPPORTED_HEAD_POSE')

    def test_hold_out_of_plane(self):
        bad=HeadObservation(.8,.5,64,0,0,0)
        self.assertEqual(self.follower.plan(bad).status,'HOLD_OUTSIDE_2D_PLANE')

    def test_hold_too_close(self):
        bad=sample(distance_eye_px=100) # approx 0.416m
        self.assertEqual(self.follower.plan(bad).status,'HOLD_SAFETY_ENVELOPE')

    def test_hold_bad_depth(self):
        bad=sample(distance_eye_px=2)
        self.assertEqual(self.follower.plan(bad).status,'HOLD_UNRELIABLE_DEPTH')


if __name__=='__main__':
    unittest.main()
