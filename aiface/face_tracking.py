"""2D face detections, temporal smoothing, ambiguous-person suppression.
No metric distance, identity or motor-control outputs are inferred.
"""
from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite

@dataclass(frozen=True)
class Face:
    x: float
    y: float
    w: float
    h: float
    confidence: float
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]

@dataclass(frozen=True)
class Observation:
    cx: float  # normalized frame coordinates [0,1]
    cy: float
    roll_deg: float  # 2D tilt, NOT 3D head pose
    confidence: float
    bbox: tuple[float, float, float, float]

@dataclass(frozen=True)
class Result:
    status: str
    observation: Observation | None
    faces_seen: int

def parse_yunet(rows) -> list[Face]:
    """OpenCV YuNet rows: x,y,w,h,right_eye,left_eye,nose,mouth,mouth,score."""
    result = []
    for row in [] if rows is None else rows:
        if len(row) < 15:
            raise ValueError('YuNet row must have 15 floats')
        vals = [float(x) for x in row[:15]]
        if all(isfinite(x) for x in vals) and vals[2] > 0 and vals[3] > 0:
            result.append(Face(*vals[:4], vals[14], (vals[4], vals[5]), (vals[6], vals[7])))
    return result

class FaceTracker:
    """One face only: returns no target if 0/2+ faces or questionable detection."""
    def __init__(self, confidence=.8, alpha=.35, max_jump=.27, max_gap=.75):
        if not (0 < confidence <= 1 and 0 < alpha <= 1 and 0 < max_jump <= 1 and max_gap > 0):
            raise ValueError('Invalid tracking settings')
        self.confidence, self.alpha, self.max_jump, self.max_gap = confidence, alpha, max_jump, max_gap
        self.last = None
        self.last_t = None

    def reset(self):
        self.last = self.last_t = None

    def update(self, faces: list[Face], width: int, height: int, now: float) -> Result:
        if width < 1 or height < 1 or not isfinite(now):
            raise ValueError('Invalid frame size/time')
        if self.last_t is not None and now < self.last_t:
            self.reset()
        if len(faces) != 1:
            self.reset()
            return Result('MULTIPLE_FACES' if faces else 'NO_FACE', None, len(faces))
        f = faces[0]
        if not 0 <= f.confidence <= 1 or f.confidence < self.confidence:
            self.reset()
            return Result('LOW_CONFIDENCE', None, 1)
        vals = (f.x, f.y, f.w, f.h, *f.right_eye, *f.left_eye)
        if not all(isfinite(x) for x in vals) or f.w <= 0 or f.h <= 0:
            self.reset()
            return Result('INVALID_FACE', None, 1)
        cx = (f.x + f.w/2) / width
        cy = (f.y + f.h/2) / height
        dx, dy = f.left_eye[0]-f.right_eye[0], f.left_eye[1]-f.right_eye[1]
        if not (0 <= cx <= 1 and 0 <= cy <= 1) or hypot(dx,dy) < 1:
            self.reset()
            return Result('INVALID_FACE', None, 1)
        roll = degrees(atan2(dy,dx))
        if roll > 90: roll -= 180
        if roll < -90: roll += 180
        obs = Observation(cx, cy, roll, f.confidence, (f.x,f.y,f.w,f.h))
        if self.last is not None and self.last_t is not None:
            if now-self.last_t > self.max_gap:
                self.reset()
            elif hypot(cx-self.last.cx, cy-self.last.cy) > self.max_jump:
                self.reset()
                return Result('REACQUIRING', None, 1)
        if self.last is not None:
            a = self.alpha
            obs = Observation(a*cx+(1-a)*self.last.cx, a*cy+(1-a)*self.last.cy,
                              a*roll+(1-a)*self.last.roll_deg, f.confidence, obs.bbox)
        self.last, self.last_t = obs, now
        return Result('TRACKING', obs, 1)
