import math
import unittest
from aiface.core import ArmConfig, MockMotion, forward, inverse, gravity_torque, mass_matrix, required_torque

C=ArmConfig()
class Kinematics(unittest.TestCase):
    def test_forward_horizontal(self):
        x,z,p=forward((0,0,0))
        self.assertAlmostEqual(x,sum(C.lengths));self.assertAlmostEqual(z,0)
        self.assertAlmostEqual(p,0)

    def test_inverse_both_elbows(self):
        for q in [(0.2,0.8,-0.1),(-0.2,-0.7,0.4),(0.4,-0.5,0.25)]:
            pose=forward(q)
            recovered=inverse(*pose,current=q)
            self.assertTrue(all(abs(a-b)<1e-7 for a,b in zip(recovered,q)))

    def test_unreachable(self):
        with self.assertRaises(ValueError): inverse(0.8,0.2,0)

    def test_no_limit_solution(self):
        with self.assertRaises(ValueError): inverse(-0.30,0.0,0.0)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError): forward((0,float('nan'),0))

class Dynamics(unittest.TestCase):
    def test_mass_matrix_symmetric(self):
        mat=mass_matrix((0.2,-0.4,0.1))
        for i in range(3):
            self.assertGreater(mat[i][i],0)
            for j in range(3): self.assertAlmostEqual(mat[i][j],mat[j][i])

    def test_horizontal_gravity(self):
        g=gravity_torque((0,0,0))
        l1,l2,l3=C.lengths;m1,m2,m3=C.masses;mp=C.phone_mass
        expected=9.80665*(m1*l1/2+m2*(l1+l2/2)+m3*(l1+l2+l3/2)+mp*(l1+l2+l3))
        self.assertAlmostEqual(g[0],expected)

    def test_vertical_gravity_zero(self):
        g=gravity_torque((math.pi/2,0,0))
        self.assertTrue(all(abs(v)<1e-10 for v in g))

    def test_torque_static(self):
        q=(0.1,-0.2,0.3)
        self.assertEqual(tuple(round(v,9) for v in required_torque(q)),
                         tuple(round(v,9) for v in gravity_torque(q)))

    def test_inertial_accel(self):
        q=(0.,0.,0.);ddq=(1.,0.,0.)
        tau=required_torque(q,(0.,0.,0.),ddq)
        m=mass_matrix(q)
        self.assertAlmostEqual(tau[0]-gravity_torque(q)[0],m[0][0])

class SafeMotion(unittest.TestCase):
    def test_converges_and_limited(self):
        motion=MockMotion();goal=inverse(0.24,0.12,0.1)
        prev=tuple(motion.q)
        for _ in range(1200):
            q=motion.step(goal,0.01)
            self.assertTrue(C.within_limits(q))
            self.assertLessEqual(max(abs(x-y) for x,y in zip(q,prev)),math.radians(C.max_speed_deg_s)*.01+1e-7)
            prev=q
        self.assertLess(max(abs(x-y) for x,y in zip(q,goal)),1e-4)

    def test_bad_dt(self):
        with self.assertRaises(ValueError): MockMotion().step((0,0,0),1)

    def test_bad_target(self):
        with self.assertRaises(ValueError): MockMotion().step((4,0,0),0.02)

    def test_halt(self):
        obj=MockMotion();obj.emergency_stop()
        self.assertEqual(obj.step((0.3,0,0),0.02),(0.,0.,0.))

    def test_invalid_config(self):
        with self.assertRaises(ValueError): ArmConfig(lengths=(0,0.1,0.1))

if __name__=='__main__': unittest.main()
