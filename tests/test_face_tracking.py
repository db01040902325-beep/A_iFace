import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from aiface.face_tracking import Face,FaceTracker,parse_yunet
from aiface.vision import verified,install_model,MODEL_SHA256

def face(x=200,y=100,confidence=.95,roll=0):
    r=math.radians(roll)
    return Face(x,y,100,100,confidence,(x+20,y+40),(x+20+50*math.cos(r),y+40+50*math.sin(r)))

class TestFaceTracking(unittest.TestCase):
    def test_yunet_parse(self):
        f=parse_yunet([[10,20,30,40,14,25,32,26,0,0,0,0,0,0,.91]])[0]
        self.assertEqual((f.x,f.w,f.confidence),(10,30,.91))
    def test_no_input(self):
        self.assertEqual(parse_yunet(None),[])
    def test_invalid_row(self):
        with self.assertRaises(ValueError):parse_yunet([[1,2]])
    def test_nan_ignored(self):
        self.assertEqual(parse_yunet([[float('nan')]+[0]*14]),[])
    def test_position_and_roll(self):
        o=FaceTracker().update([face(200,100,roll=15)],640,480,1).observation
        self.assertAlmostEqual(o.cx,250/640)
        self.assertAlmostEqual(o.cy,150/480)
        self.assertAlmostEqual(o.roll_deg,15)
    def test_smoothing(self):
        t=FaceTracker(alpha=.5)
        a=t.update([face(100)],640,480,1).observation
        b=t.update([face(140)],640,480,1.02).observation
        self.assertAlmostEqual(b.cx,(a.cx+190/640)/2)
    def test_no_face(self):
        t=FaceTracker();t.update([face()],640,480,1)
        r=t.update([],640,480,1.02)
        self.assertEqual(r.status,'NO_FACE');self.assertIsNone(r.observation)
    def test_multiple_faces(self):
        r=FaceTracker().update([face(),face(400)],640,480,1)
        self.assertEqual(r.status,'MULTIPLE_FACES');self.assertIsNone(r.observation)
    def test_low_confidence(self):
        self.assertEqual(FaceTracker().update([face(confidence=.6)],640,480,1).status,'LOW_CONFIDENCE')
    def test_jump_rejected(self):
        t=FaceTracker(max_jump=.1);t.update([face(10)],640,480,1)
        self.assertEqual(t.update([face(400)],640,480,1.02).status,'REACQUIRING')
    def test_gap_reacquires(self):
        t=FaceTracker(max_jump=.1);t.update([face(10)],640,480,1)
        self.assertEqual(t.update([face(400)],640,480,3).status,'TRACKING')
    def test_bad_frame(self):
        with self.assertRaises(ValueError):FaceTracker().update([face()],0,480,1)
    def test_bad_eyes(self):
        f=Face(1,2,100,100,.9,(3,3),(3,3))
        self.assertEqual(FaceTracker().update([f],640,480,1).status,'INVALID_FACE')
    def test_model_hash(self):
        self.assertEqual(len(MODEL_SHA256),64)
        with TemporaryDirectory() as d:
            p=Path(d)/'file.onnx';p.write_text('bad')
            self.assertFalse(verified(p))
    def test_download_mismatch(self):
        class MockResponse:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):return b'WRONG'
        with TemporaryDirectory() as d:
            p=Path(d)/'file.onnx'
            with patch('aiface.vision.urlopen',return_value=MockResponse()):
                with self.assertRaises(ValueError):install_model(p)
            self.assertFalse(p.exists())

if __name__=='__main__':unittest.main()
