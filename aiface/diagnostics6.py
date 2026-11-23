"""Diagnose six-axis *simulated* face-follow accuracy from existing webcam CSV.

Separates initial/reacquisition tracking (controller catch-up) from sustained
tracking, then groups sustained frames by observed change of the *target* pose.
A changing target can be actual face motion OR monocular camera noise: this
log alone cannot distinguish them. No physical robot or motor metrics.
"""
import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

TRACK = 'TRACK_SIMULATION_6R'
AXES = ('x','y','z')


def _stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        'count': len(values),
        'mean': statistics.fmean(values),
        'max': ordered[-1],
        'p95': ordered[math.ceil(.95*len(ordered))-1],
    }


def analyze(path, *, startup_seconds=1.56, low_change_mm_s=20.):
    """Return per-phase accuracy and target-change diagnostics.

    startup_seconds comes from a previous synthetic 6-cm step response and
    is only a comparison window, not a guaranteed physical settling time.
    """
    if not math.isfinite(startup_seconds) or startup_seconds < 0:
        raise ValueError('Invalid startup comparison window')
    if not math.isfinite(low_change_mm_s) or low_change_mm_s <= 0:
        raise ValueError('Invalid target-change threshold')

    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'time_s','status'} | {
            f'{prefix}_{axis}_m'
            for prefix in ('target','actual') for axis in AXES
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {sorted(required-set(reader.fieldnames or []))}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty CSV')

    errors = {
        'all': [], 'startup': [], 'continuous': [],
        'low_change': [], 'high_change': [],
    }
    axes = {'x': [],'y': [],'z': []}
    delta_target = []
    target_speed = []
    dt_values = []
    status_counts = Counter()
    segments = 0
    malformed = 0
    ignored_intervals = 0
    previous_time = None
    prev_track = None
    segment_start = None

    for index, row in enumerate(rows, 1):
        try:
            t=float(row['time_s'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid time at CSV row {index}') from exc
        if not math.isfinite(t) or (previous_time is not None and t <= previous_time):
            raise ValueError(f'Time is non-finite or non-increasing at row {index}')
        previous_time = t
        status=row['status']
        status_counts[status]+=1
        if status != TRACK:
            segment_start=None
            prev_track=None
            continue
        try:
            target=tuple(float(row[f'target_{axis}_m']) for axis in AXES)
            actual=tuple(float(row[f'actual_{axis}_m']) for axis in AXES)
        except (TypeError, ValueError):
            malformed+=1
            segment_start=None
            prev_track=None
            continue
        if not all(math.isfinite(v) for v in target+actual):
            malformed+=1
            segment_start=None
            prev_track=None
            continue

        if segment_start is None:
            segment_start=t
            segments+=1
        diff=tuple(1000*(actual[i]-target[i]) for i in range(3))
        error=math.sqrt(sum(v*v for v in diff))
        errors['all'].append(error)
        for axis,delta in zip(AXES,diff):
            axes[axis].append(abs(delta))

        if t-segment_start <= startup_seconds:
            errors['startup'].append(error)
        else:
            errors['continuous'].append(error)
            if prev_track is not None:
                prior_t,prior_target=prev_track
                dt=t-prior_t
                if dt <= .25:
                    speed_mm_s=math.dist(target,prior_target)*1000/dt
                    step_mm=math.dist(target,prior_target)*1000
                    target_speed.append(speed_mm_s)
                    delta_target.append(step_mm)
                    dt_values.append(dt)
                    key='low_change' if speed_mm_s <= low_change_mm_s else 'high_change'
                    errors[key].append(error)
                else:
                    ignored_intervals+=1
        prev_track=(t,target)

    return {
        'total_frames':len(rows),
        'status_counts':status_counts,
        'tracking_segments':segments,
        'valid_track_frames':len(errors['all']),
        'invalid_track_frames':malformed,
        'groups':{k:_stats(v) for k,v in errors.items()},
        'axis_abs_mean_mm':{axis:statistics.fmean(a) if a else None
                            for axis,a in axes.items()},
        'target_step_mm':_stats(delta_target),
        'target_speed_mm_s':_stats(target_speed),
        'frame_interval_ms':_stats([1000*v for v in dt_values]),
        'skipped_large_intervals':ignored_intervals,
        'startup_seconds':startup_seconds,
        'low_change_mm_s':low_change_mm_s,
    }


def format_report(result):
    lines=[
        '===== A_iFace 6R 추적 오차 원인 분리 (CSV) =====',
        f'전체 프레임: {result["total_frames"]}',
        f'정상 추적 프레임: {result["valid_track_frames"]}',
        f'추적 구간(재시작 포함): {result["tracking_segments"]}',
        f'분석 불가 추적 행: {result["invalid_track_frames"]}',
        '',
        '[구간별 위치 오차: 목표 좌표 대비 가상 관절 말단, 단위 mm]',
    ]
    descriptions=(
        ('all','전체 추적'),
        ('startup',f'시작·재인식 직후 {result["startup_seconds"]:.2f}초'),
        ('continuous',f'{result["startup_seconds"]:.2f}초 이후 연속 추적'),
        ('low_change',f'연속 구간 목표 변화 적음 (≤{result["low_change_mm_s"]:.1f} mm/s)'),
        ('high_change',f'연속 구간 목표 변화 큼 (>{result["low_change_mm_s"]:.1f} mm/s)'),
    )
    for key,name in descriptions:
        s=result['groups'][key]
        if not s:
            lines.append(f'{name}: 관측 없음')
        else:
            lines.append(f'{name}: {s["count"]}프레임, 평균 {s["mean"]:.2f}, '
                         f'95백분위 {s["p95"]:.2f}, 최대 {s["max"]:.2f}')
    lines.append('')
    lines.append('[축별 평균 절대오차]')
    for axis in AXES:
        v=result['axis_abs_mean_mm'][axis]
        lines.append(f'{axis.upper()}: {v:.2f} mm' if v is not None
                     else f'{axis.upper()}: 관측 없음')
    lines.append('')
    lines.append('[연속 추적 목표 좌표의 프레임 간 변화]')
    for field,name,unit in (
        ('target_step_mm','목표 위치 변화량','mm/프레임'),
        ('target_speed_mm_s','목표 위치 변화 속도','mm/s'),
        ('frame_interval_ms','프레임 간격','ms')
    ):
        stat=result[field]
        lines.append(f'{name}: 평균 {stat["mean"]:.2f} {unit}, '
                     f'95백분위 {stat["p95"]:.2f} {unit}'
                     if stat else f'{name}: 비교 가능한 연속 프레임 없음')
    lines.append(f'0.25초 초과 불연속 구간 제외: {result["skipped_large_intervals"]}개')
    lines.extend((
        '',
        '해석: 시작 구간 오차가 크면 초기·재인식 후 이동 영향 가능성이 있습니다.',
        '연속 구간에도 오차가 크다면 목표 변화와 제어 지연을 추가 분석해야 합니다.',
        '목표 변화량에는 실제 얼굴 이동과 영상 측정 잡음이 모두 포함됩니다.',
        '이 CSV만으로 둘을 구분하거나 실제 모터 성능을 평가할 수 없습니다.',
    ))
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description='6R tracked target vs virtual FK accuracy')
    p.add_argument('csv_path',type=Path,help='e.g. results/face6_live.csv')
    p.add_argument('--startup-seconds',type=float,default=1.56)
    p.add_argument('--low-change-mm-s',type=float,default=20.)
    a=p.parse_args()
    print(format_report(analyze(
        a.csv_path,startup_seconds=a.startup_seconds,
        low_change_mm_s=a.low_change_mm_s
    )))


if __name__=='__main__':
    main()
