"""6R 3D FK checks independent of the existing 3R simulation."""
import math
import unittest

from aiface.arm6 import Arm6Config, forward6
from aiface.core import forward as forward3


class Forward6Tests(unittest.TestCase):
    def test_zero_pose(self):
        pose = forward6((0.,)*6)
        self.assertAlmostEqual(pose.position_m[0], .32)
        self.assertAlmostEqual(pose.position_m[1], 0.)
        self.assertAlmostEqual(pose.position_m[2], .12)
        self.assertEqual(len(pose.points_m), 4)

    def test_base_yaw_rotates_lateral_position(self):
        q = (math.pi/2, 0., 0., 0., 0., 0.)
        pose = forward6(q)
        self.assertAlmostEqual(pose.position_m[0], 0., places=12)
        self.assertAlmostEqual(pose.position_m[1], .32)
        self.assertAlmostEqual(pose.position_m[2], .12)

    def test_shoulder_pitch_lifts_link(self):
        p = forward6((0., math.pi/6, 0., 0., 0., 0.))
        self.assertAlmostEqual(p.position_m[0], .32*math.cos(math.pi/6))
        self.assertAlmostEqual(p.position_m[2], .12 + .32*math.sin(math.pi/6))

    def test_planar_3r_matches_at_zero_base_yaw(self):
        q3 = (.2, -.4, .3)
        x, z, phi = forward3(q3)
        p = forward6((0., *q3, 0., 0.))
        self.assertAlmostEqual(p.position_m[0], x)
        self.assertAlmostEqual(p.position_m[1], 0.)
        self.assertAlmostEqual(p.position_m[2]-.12, z)
        self.assertAlmostEqual(p.rotation_matrix[0][0], math.cos(phi))
        self.assertAlmostEqual(p.rotation_matrix[2][0], math.sin(phi))

    def test_wrist_only_changes_orientation(self):
        base = forward6((0.,)*6)
        wrist = forward6((0.,0.,0.,0.,math.pi/2,math.pi/3))
        self.assertEqual(base.position_m, wrist.position_m)
        self.assertNotEqual(base.rotation_matrix, wrist.rotation_matrix)

    def test_rotation_is_orthonormal(self):
        q = tuple(map(math.radians, (45, 30, -25, -5, 15, 20)))
        R = forward6(q).rotation_matrix
        for i in range(3):
            for j in range(3):
                dot = sum(R[k][i]*R[k][j] for k in range(3))
                self.assertAlmostEqual(dot, float(i==j), places=12)
        det = (R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])
               -R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])
               +R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
        self.assertAlmostEqual(det,1.,places=12)

    def test_limits_and_nan_rejected(self):
        for bad in ((0.,)*5, (math.nan,)*6,
                    (math.radians(171),0.,0.,0.,0.,0.)):
            with self.assertRaises(ValueError):
                forward6(bad)

    def test_bad_parameters_rejected(self):
        for kwargs in ({"lengths_m":(0.,.1,.1)},
                       {"base_height_m":-1.},
                       {"limits_deg":((0.,1.),)*5}):
            with self.assertRaises(ValueError):
                Arm6Config(**kwargs)


if __name__=="__main__":
    unittest.main()
