"""Diagnostics from existing 6R webcam tracking CSV, without camera access."""
import csv
import math
import tempfile
import unittest
from pathlib import Path

from aiface.diagnostics6 import analyze, format_report


FIELDS=['time_s','status',
        'target_x_m','target_y_m','target_z_m',
        'actual_x_m','actual_y_m','actual_z_m']


class Diagnose6Tests(unittest.TestCase):
    def save(self,rows):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out=Path(tmp.name)/'follow6.csv'
        with out.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        return out

    def row(self,t,status='TRACK_SIMULATION_6R',target=.20,actual=.20):
        return {
            'time_s':t,'status':status,
            'target_x_m':target,'target_y_m':0.,'target_z_m':.12,
            'actual_x_m':actual,'actual_y_m':0.,'actual_z_m':.12,
        }

    def test_transient_vs_continuous_by_episode(self):
        out=self.save([
            self.row(.02,actual=.10), # 100mm transient
            self.row(.52,actual=.17), # 30mm transient
            self.row(2.02,actual=.20), # 0mm continuous
            self.row(2.10,'HOLD_NO_FACE'),
            self.row(3.00,actual=.13), # 70mm transient after reacquisition
            self.row(4.70,actual=.19), # 10mm continuous
        ])
        result=analyze(out)
        self.assertEqual(result['total_frames'],6)
        self.assertEqual(result['valid_track_frames'],5)
        self.assertEqual(result['tracking_segments'],2)
        self.assertEqual(result['groups']['startup']['count'],3)
        self.assertEqual(result['groups']['continuous']['count'],2)
        self.assertAlmostEqual(result['groups']['startup']['mean'],200/3)
        self.assertAlmostEqual(result['groups']['continuous']['mean'],5)
        self.assertIn('추적 구간',format_report(result))

    def test_continuous_target_speed_categories(self):
        out=self.save([
            self.row(.01,target=.20,actual=.10),
            self.row(.51,target=.20,actual=.15),
            self.row(1.70,target=.20,actual=.19),
            self.row(1.80,target=.201,actual=.20), # 10 mm/s -> low
            self.row(1.90,target=.211,actual=.21), # 100 mm/s -> high
        ])
        result=analyze(out)
        self.assertEqual(result['groups']['continuous']['count'],3)
        self.assertEqual(result['groups']['low_change']['count'],1)
        self.assertEqual(result['groups']['high_change']['count'],1)
        self.assertEqual(result['target_speed_mm_s']['count'],2)

    def test_missing_values_do_not_fake_zero_accuracy(self):
        bad=self.row(0)
        bad['actual_x_m']=''
        out=self.save([bad,self.row(.02)])
        r=analyze(out)
        self.assertEqual(r['invalid_track_frames'],1)
        self.assertEqual(r['valid_track_frames'],1)

    def test_only_hold_reports_no_tracked_frames(self):
        out=self.save([self.row(.1,'HOLD_NO_FACE'),
                       self.row(.2,'HOLD_MULTIPLE_FACES')])
        result=analyze(out)
        self.assertIsNone(result['groups']['all'])
        self.assertIn('관측 없음',format_report(result))
        self.assertEqual(result['tracking_segments'],0)

    def test_bad_timestamp_rejected(self):
        for times in ((.2,.1),(.1,.1),(.1,float('nan'))):
            with self.subTest(times=times):
                out=self.save([self.row(t) for t in times])
                with self.assertRaisesRegex(ValueError,'Time'):
                    analyze(out)

    def test_invalid_parameters_rejected(self):
        out=self.save([self.row(.1)])
        for kwargs in ({'startup_seconds':-1},
                       {'low_change_mm_s':0},
                       {'startup_seconds':math.nan}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    analyze(out,**kwargs)

    def test_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.csv'
            path.write_text('time_s,status\n1,TRACK_SIMULATION_6R\n')
            with self.assertRaisesRegex(ValueError,'Missing columns'):
                analyze(path)


if __name__=='__main__':
    unittest.main()
