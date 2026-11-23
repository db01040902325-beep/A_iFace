"""Six-axis mock trajectory, CSV, and software halt regression."""
import csv
import math
import tempfile
import unittest
from pathlib import Path

from aiface.arm6 import Arm6Config, forward6
from aiface.motion6 import MockMotion6, demo
from tools.check_motion6_limits import check


def r(deg):
    return tuple(math.radians(v) for v in deg)


class SixAxisMotionTests(unittest.TestCase):
    def verify_samples(self, targets, dt=.02):
        cfg=Arm6Config()
        arm=MockMotion6(cfg)
        prev_q=(0.,)*6
        prev_v=(0.,)*6
        max_accel=math.radians(cfg.max_accel_deg_s2)
        max_speed=math.radians(cfg.max_speed_deg_s)
        for target in targets:
            q=arm.step(target,dt)
            v=tuple((q[j]-prev_q[j])/dt for j in range(6))
            self.assertTrue(cfg.within_limits(q))
            for j in range(6):
                self.assertLessEqual(abs(v[j]),max_speed+1e-8)
                self.assertLessEqual(abs((v[j]-prev_v[j])/dt),max_accel+1e-7)
            prev_q,prev_v=q,v
        return arm

    def test_all_joints_converge(self):
        goal=r((65.,-40.,105.,-95.,85.,-120.))
        arm=self.verify_samples([goal]*1000)
        self.assertLess(max(abs(a-b) for a,b in zip(arm.q,goal)),1e-5)
        self.assertLess(max(map(abs,arm.dq)),1e-6)

    def test_retarget_mid_motion(self):
        a=r((75.,55.,-80.,60.,-60.,90.))
        b=r((-65.,-60.,75.,-60.,60.,-95.))
        arm=self.verify_samples([a]*140+[b]*240+[(0.,)*6]*600)
        self.assertLess(max(map(abs,arm.q)),1e-5)

    def test_variable_dt_and_small_moves(self):
        goal=r((.001,-.002,.003,-.004,.005,-.006))
        for dt in (.01,.02,.05,.1):
            with self.subTest(dt=dt):
                arm=self.verify_samples([goal]*70,dt)
                self.assertLess(max(abs(a-b) for a,b in zip(arm.q,goal)),1e-7)

    def test_provisional_joint_limits(self):
        goal=r(tuple(hi for lo,hi in Arm6Config().limits_deg))
        arm=self.verify_samples([goal]*1400)
        self.assertLess(max(abs(a-b) for a,b in zip(arm.q,goal)),1e-5)

    def test_software_emergency_stop_and_explicit_reset(self):
        arm=MockMotion6()
        goal=r((30.,-35.,25.,-15.,30.,-40.))
        for _ in range(50):
            arm.step(goal,.02)
        stopped=tuple(arm.q)
        arm.emergency_stop()
        self.assertTrue(arm.halted)
        self.assertEqual(arm.dq,[0.]*6)
        for _ in range(100):
            self.assertEqual(arm.step(goal,.02),stopped)
        arm.reset()
        self.assertFalse(arm.halted)
        self.assertNotEqual(arm.step(goal,.02),stopped)

    def test_reject_invalid_target_and_dt(self):
        arm=MockMotion6()
        invalid=(
            ((0.,)*5, .02),
            (r((180.,0,0,0,0,0)), .02),
            ((math.nan,)*6,.02),
            ((0.,)*6,0.),
            ((0.,)*6,.11),
            ((0.,)*6,math.nan),
        )
        for goal, dt in invalid:
            with self.subTest(goal=goal,dt=dt):
                with self.assertRaises(ValueError):
                    arm.step(goal,dt)

    def test_fk_updates_with_base_yaw(self):
        arm=MockMotion6()
        start=forward6(tuple(arm.q))
        for _ in range(160):
            arm.step(r((45,0,0,0,0,0)),.02)
        end=forward6(tuple(arm.q))
        self.assertGreater(end.position_m[1],.01)
        self.assertLess(end.position_m[0],start.position_m[0])

    def test_csv_and_postprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'motion6.csv'
            demo(path)
            with path.open(newline='',encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),810)
            self.assertEqual({r['phase'] for r in rows},
                             {'approach','retarget','home'})
            check(path)

    def test_bad_config_parameters(self):
        for kwargs in ({'max_speed_deg_s':0.},
                       {'max_accel_deg_s2':-1.},
                       {'max_speed_deg_s':math.nan}):
            with self.assertRaises(ValueError):
                Arm6Config(**kwargs)


if __name__=='__main__':
    unittest.main()
