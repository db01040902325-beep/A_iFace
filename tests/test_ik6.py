"""6R inverse kinematics regression and fail-closed cases."""
import math
import random
import unittest

from aiface.arm6 import Arm6Config, forward6
from aiface.ik6 import inverse6


def rad(degrees):
    return tuple(map(math.radians, degrees))


class Inverse6Tests(unittest.TestCase):
    def assert_roundtrip(self, angles):
        target = forward6(angles)
        found = inverse6(target.position_m, target.rotation_matrix, current=angles)
        self.assertTrue(Arm6Config().within_limits(found))
        result = forward6(found)
        for v, t in zip(result.position_m, target.position_m):
            self.assertAlmostEqual(v, t, delta=1e-7)
        for r1, r2 in zip(result.rotation_matrix, target.rotation_matrix):
            for v, t in zip(r1, r2):
                self.assertAlmostEqual(v, t, delta=1e-7)
        for v, t in zip(found, angles):
            self.assertAlmostEqual(v, t, delta=1e-5)

    def test_all_zero_pose(self):
        self.assert_roundtrip((0.,)*6)

    def test_base_yaw_90_and_mixed_pose(self):
        for deg in ((90, 0, 0, 0, 0, 0),
                    (45,30,-25,-5,15,20),
                    (-70,-30,55,-25,-20,-35),
                    (160,60,120,80,140,-165),
                    (-160,-60,-120,-80,-135,160)):
            with self.subTest(deg=deg):
                self.assert_roundtrip(rad(deg))

    def test_200_random_reachable_poses(self):
        rng = random.Random(20261123)
        for i in range(200):
            deg = (rng.uniform(-169,169), rng.uniform(-84,84),
                   rng.uniform(-144,144), rng.uniform(-144,144),
                   rng.uniform(-159,159), rng.uniform(-169,169))
            with self.subTest(case=i):
                self.assert_roundtrip(rad(deg))

    def test_returns_closest_elbow_branch(self):
        a = rad((0, 15, 60, -30, 10, 5))
        p = forward6(a)
        b = inverse6(p.position_m, p.rotation_matrix,current=a)
        self.assertLess(max(abs(x-y) for x,y in zip(a,b)), 1e-5)

    def test_invalid_rotation_rejected(self):
        identity = forward6((0.,)*6).rotation_matrix
        bad = (
            ((1,0,0),(0,1,0),(0,0,-1)),
            ((1,0,0),(0,1,0),(0,0,2)),
            ((1,0,0),(0,1,0)),
            ((float('nan'),0,0),(0,1,0),(0,0,1)),
        )
        for R in bad:
            with self.subTest(R=R), self.assertRaises(ValueError):
                inverse6((.32,0,.12), R)
        with self.assertRaises(ValueError):
            inverse6((float('nan'),0,0),identity)
        with self.assertRaises(ValueError):
            inverse6((.2,0,.12),identity,current=(0.,)*5)

    def test_unreachable_goal_rejected(self):
        identity = forward6((0.,)*6).rotation_matrix
        with self.assertRaises(ValueError):
            inverse6((1.,1.,1.),identity)

    def test_axis_singularity_rejected(self):
        identity = forward6((0.,)*6).rotation_matrix
        with self.assertRaisesRegex(ValueError, 'Base yaw singularity'):
            inverse6((0.,0.,.2),identity)

    def test_wrist_roll_singularity_rejected(self):
        p = forward6(rad((30,10,30,-5,90,20)))
        with self.assertRaisesRegex(ValueError, 'No nonsingular IK solution'):
            inverse6(p.position_m,p.rotation_matrix)

    def test_limit_rejected(self):
        p = forward6(rad((90,10,30,-5,20,20)))
        cfg=Arm6Config(limits_deg=((-10,10),(-85,85),(-145,145),(-145,145),(-160,160),(-170,170)))
        with self.assertRaises(ValueError):
            inverse6(p.position_m,p.rotation_matrix,cfg=cfg)


if __name__ == '__main__':
    unittest.main()
