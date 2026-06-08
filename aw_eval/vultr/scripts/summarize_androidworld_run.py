#!/usr/bin/env python3
from __future__ import annotations
import gzip, json, math, numbers, pickle, pathlib, statistics, sys, time


def mean(xs):
    xs=[float(x) for x in xs if isinstance(x, numbers.Number) and math.isfinite(float(x))]
    return statistics.mean(xs) if xs else None


def finite_number(value):
    return isinstance(value, numbers.Number) and math.isfinite(float(value))


def load_episode_file(path: pathlib.Path):
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as exc:
        return [{"_load_error": f"{type(exc).__name__}:{exc}", "_file": str(path)}]


def model_from_path(path: pathlib.Path, root: pathlib.Path):
    rel = path.relative_to(root)
    first = rel.parts[0] if rel.parts else "unknown"
    if first.startswith('base_'):
        return 'base'
    if first.startswith('current_'):
        return 'current'
    return first.split('_')[0]


def as_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def normalize_episode_data(value):
    """Return per-step dict records from known AndroidWorld episode_data variants."""
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        list_lens=[len(v) for v in value.values() if isinstance(v, list)]
        if list_lens:
            rows=[]
            for i in range(max(list_lens)):
                row={}
                for k, v in value.items():
                    row[k]=v[i] if isinstance(v, list) and i < len(v) else (None if isinstance(v, list) else v)
                rows.append(row)
            return rows
        return [value]
    return []


def collect_step_values(steps, key):
    return [step.get(key) for step in steps if isinstance(step, dict) and step.get(key) is not None]


def main():
    if len(sys.argv) < 2:
        print('usage: summarize_androidworld_run.py RUN_ROOT', file=sys.stderr)
        raise SystemExit(2)
    root=pathlib.Path(sys.argv[1]).resolve()
    episodes=[]
    for f in sorted(root.rglob('*.pkl.gz')):
        model=model_from_path(f, root)
        for ep in load_episode_file(f):
            steps=normalize_episode_data(ep.get('episode_data') if isinstance(ep, dict) else None)
            pred=[]
            for item in collect_step_values(steps, 'server_prediction'):
                pred.extend(as_list(item))
            structural=[]
            for item in collect_step_values(steps, 'structural'):
                structural.extend(as_list(item))
            lat=[x.get('latency_ms') for x in pred if isinstance(x,dict) and isinstance(x.get('latency_ms'),(int,float))]
            total_lat=collect_step_values(steps, 'latency_total_ms')
            strict=sum(1 for s in structural if isinstance(s,dict) and s.get('valid_json'))
            valid=sum(1 for s in structural if isinstance(s,dict) and s.get('valid_mobile_use'))
            repaired=sum(1 for s in structural if isinstance(s,dict) and s.get('repaired'))
            nstruct=len(structural)
            episodes.append({
                'model': model,
                'task': ep.get('task_template') if isinstance(ep,dict) else None,
                'goal': ep.get('goal') if isinstance(ep,dict) else None,
                'success': ep.get('is_successful') if isinstance(ep,dict) else None,
                'episode_length': ep.get('episode_length') if isinstance(ep,dict) else None,
                'run_time': ep.get('run_time') if isinstance(ep,dict) else None,
                'exception_info': str(ep.get('exception_info')) if isinstance(ep,dict) and ep.get('exception_info') is not None else None,
                'strict_json_rate': strict / nstruct if nstruct else None,
                'mobile_use_rate': valid / nstruct if nstruct else None,
                'repair_rate': repaired / nstruct if nstruct else None,
                'model_latency_ms_mean': mean(lat),
                'total_latency_ms_mean': mean(total_lat),
                'file': str(f),
            })
    by_model={}
    for ep in episodes:
        m=ep['model']; by_model.setdefault(m, []).append(ep)
    aggregate={}
    for m, rows in by_model.items():
        succ=[float(r['success']) for r in rows if finite_number(r.get('success'))]
        aggregate[m]={
            'episodes': len(rows),
            'success_count': sum(1 for x in succ if x > 0),
            'success_rate': mean(succ),
            'strict_json_rate_mean': mean([r.get('strict_json_rate') for r in rows]),
            'mobile_use_rate_mean': mean([r.get('mobile_use_rate') for r in rows]),
            'repair_rate_mean': mean([r.get('repair_rate') for r in rows]),
            'model_latency_ms_mean': mean([r.get('model_latency_ms_mean') for r in rows]),
            'total_latency_ms_mean': mean([r.get('total_latency_ms_mean') for r in rows]),
            'episode_run_time_mean': mean([r.get('run_time') for r in rows]),
        }
    out={'root': str(root), 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'aggregate': aggregate, 'episodes': episodes}
    (root/'summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    with (root/'episodes.jsonl').open('w', encoding='utf-8') as f:
        for row in episodes:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(aggregate, indent=2))
    print('WROTE', root/'summary.json')

if __name__ == '__main__':
    main()
