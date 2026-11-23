"""No-camera verification of 6-axis face following and safety holds."""
import csv
import math
import tempfile
import unittest
from pathlib import Path

from aiface.arm6 import forward6
from aiface.follow6 import FaceFollower6, SimulatedFace6Controller, synthetic_demo
from aiface.head_pose import HeadObservation, MonoCalibration, FixedCameraRig


def sample(*, cx=.5, cy=.5, distance=.65, pitch=0., yaw=0., roll=0.):
    return HeadObservation(cx,cy,64*.65/distance,pitch,yaw,roll)


class Face6PlanTests(unittest.TestCase):
    def setUp(self):
        self.f=FaceFollower6(MonoCalibration(.65,64),FixedCameraRig())

    def test_center_target_and_six_axes(self):
        p=self.f.plan(sample())
        self.assertEqual(p.status,'TRACK_SIMULATION_6R')
        self.assertEqual(len(p.target_joint_angles),6)
        self.assertAlmostEqual(p.distance_m,.65)
        self.assertAlmostEqual(p.target_xyz_m[0],.20)
        self.assertAlmostEqual(p.target_xyz_m[1],0.)
        self.assertAlmostEqual(p.target_xyz_m[2],.12)
        result=forward6(p.target_joint_angles)
        for x,y in zip(result.position_m,p.target_xyz_m):
            self.assertAlmostEqual(x,y,delta=1e-8)

    def test_lateral_left_right_targets_use_base_yaw(self):
        left=self.f.plan(sample(cx=.4))
        right=self.f.plan(sample(cx=.6))
        self.assertEqual(left.status,'TRACK_SIMULATION_6R')
        self.assertEqual(right.status,'TRACK_SIMULATION_6R')
        self.assertLess(left.target_xyz_m[1],0.)
        self.assertGreater(right.target_xyz_m[1],0.)
        self.assertLess(left.target_joint_angles[0],0.)
        self.assertGreater(right.target_joint_angles[0],0.)

    def test_height_and_depth(self):
        high=self.f.plan(sample(cy=.45,distance=.68))
        low=self.f.plan(sample(cy=.55,distance=.68))
        self.assertEqual(high.status,'TRACK_SIMULATION_6R')
        self.assertEqual(low.status,'TRACK_SIMULATION_6R')
        self.assertGreater(high.target_xyz_m[2],low.target_xyz_m[2])
        near=self.f.plan(sample(distance=.62))
        far=self.f.plan(sample(distance=.68))
        self.assertEqual(near.status,'TRACK_SIMULATION_6R')
        self.assertEqual(far.status,'TRACK_SIMULATION_6R')
        self.assertLess(near.target_xyz_m[0],far.target_xyz_m[0])

    def test_head_orientation_changes_wrist_commands(self):
        base=self.f.plan(sample())
        turned=self.f.plan(sample(pitch=4,yaw=5,roll=3))
        self.assertEqual(turned.status,'TRACK_SIMULATION_6R')
        self.assertNotEqual(base.target_joint_angles,turned.target_joint_angles)
        a=forward6(base.target_joint_angles)
        b=forward6(turned.target_joint_angles)
        self.assertNotEqual(a.rotation_matrix,b.rotation_matrix)

    def test_no_face_many_face_calibration(self):
        self.assertEqual(self.f.plan(None,faces=0).status,'HOLD_NO_FACE')
        self.assertEqual(self.f.plan(sample(),faces=2).status,'HOLD_MULTIPLE_FACES')
        self.assertEqual(self.f.plan(None).status,'HOLD_NO_FACE_MATRIX')
        missing=FaceFollower6(None,FixedCameraRig())
        self.assertEqual(missing.plan(sample()).status,'HOLD_NEEDS_CALIBRATION')

    def test_unreliable_or_unsafe(self):
        self.assertEqual(self.f.plan(sample(yaw=23)).status,
                         'HOLD_UNSUPPORTED_HEAD_POSE')
        self.assertEqual(self.f.plan(sample(distance=.42)).status,
                         'HOLD_SAFETY_ENVELOPE')
        self.assertEqual(self.f.plan(sample(distance=2.0)).status,
                         'HOLD_UNRELIABLE_DEPTH')
        self.assertEqual(self.f.plan(sample(cx=float('nan'))).status,
                         'HOLD_INVALID_OBSERVATION')
        self.assertEqual(self.f.plan(sample(cx=.99)).status,
                         'HOLD_IK_UNREACHABLE')

    def test_never_trust_uncalibrated_rig(self):
        with self.assertRaises(ValueError):
            FaceFollower6(MonoCalibration(.65,64),FixedCameraRig(),desired_standoff_m=.2)


class Face6ControllerTests(unittest.TestCase):
    def setUp(self):
        self.ctrl=SimulatedFace6Controller(
            FaceFollower6(MonoCalibration(.65,64),FixedCameraRig())
        )

    def follow(self,obs,frames=1,faces=1):
        plan=None
        for _ in range(frames):
            plan=self.ctrl.update(obs,dt=.02,faces=faces)
        return plan

    def test_reacquire_and_hold_freezes_all_six_joints(self):
        obs=sample(cx=.53)
        self.assertEqual(self.follow(obs,4).status,'HOLD_REACQUIRING')
        self.assertEqual(self.follow(obs).status,'TRACK_SIMULATION_6R')
        self.follow(obs,60)
        previous=tuple(self.ctrl.motion.q)
        self.assertEqual(self.follow(None,5,0).status,'HOLD_NO_FACE')
        self.assertEqual(previous,tuple(self.ctrl.motion.q))
        self.assertEqual(self.follow(obs,4).status,'HOLD_REACQUIRING')
        self.assertEqual(previous,tuple(self.ctrl.motion.q))
        self.assertEqual(self.follow(obs).status,'TRACK_SIMULATION_6R')
        self.assertNotEqual(previous,tuple(self.ctrl.motion.q))

    def test_multiple_faces_holds(self):
        obs=sample()
        self.follow(obs,60)
        p=tuple(self.ctrl.motion.q)
        self.assertEqual(self.follow(obs,10,2).status,'HOLD_MULTIPLE_FACES')
        self.assertEqual(p,tuple(self.ctrl.motion.q))

    def test_manual_stop_latched_until_reset(self):
        obs=sample(cx=.54)
        self.follow(obs,25)
        self.ctrl.manual_stop()
        pos=tuple(self.ctrl.motion.q)
        self.assertEqual(self.follow(obs,20).status,'HOLD_MANUAL_STOP')
        self.assertEqual(pos,tuple(self.ctrl.motion.q))
        self.ctrl.reset_manual()
        self.assertEqual(self.follow(obs,4).status,'HOLD_REACQUIRING')
        self.assertEqual(pos,tuple(self.ctrl.motion.q))
        self.assertEqual(self.follow(obs).status,'TRACK_SIMULATION_6R')

    def test_malformed_dt_fail_closed(self):
        obs=sample()
        self.follow(obs,15)
        p=tuple(self.ctrl.motion.q)
        self.assertEqual(self.ctrl.update(obs,dt=float('nan')).status,
                         'HOLD_INVALID_DT')
        self.assertEqual(p,tuple(self.ctrl.motion.q))

    def test_synthetic_demo_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'follow6.csv'
            synthetic_demo(out)
            with out.open(newline='',encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),400)
            self.assertTrue(any(r['status']=='TRACK_SIMULATION_6R' for r in rows))
            self.assertTrue(any(r['status']=='HOLD_NO_FACE' for r in rows))
            self.assertTrue(any(r['status']=='HOLD_MULTIPLE_FACES' for r in rows))
            for prev,cur in zip(rows,rows[1:]):
                if cur['status'] != 'TRACK_SIMULATION_6R':
                    self.assertTrue(all(
                        float(prev[f'q{j}_deg'])==float(cur[f'q{j}_deg'])
                        for j in range(1,7)))


if __name__ == '__main__':
    unittest.main()
