#!/usr/bin/env python3
"""Execute the extracted notebook cells sequentially in one Python namespace.

Install requirements before running. This reproduces the original notebook's
shared-variable execution model; individual step scripts are not independent CLIs.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='MS-XAI-Load-Opt chronological experiment runner')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'data', help='Input CSV folder')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'results', help='Output folder')
    parser.add_argument('--check-only', action='store_true', help='Check input-file presence and Python source syntax without executing experiments')
    parser.add_argument('--stop-after-step', type=int, choices=range(2,33), default=32, help='Stop after this Python step; default 32 (complete pipeline)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    required = [data_dir / 'hourly_energy_data.csv', data_dir / 'owid_energy_data.csv']
    missing = [p for p in required if not p.is_file()]
    step_files = sorted((root / 'steps').glob('*.py'))
    assert len(step_files) == 31, f'Expected 31 Python steps, found {len(step_files)}'
    for p in step_files:
        compile(p.read_text(encoding='utf-8'), str(p), 'exec')
    print(f'Source validation successful: {len(step_files)} Python steps')
    print(f'Input folder: {data_dir}')
    print(f'Output folder: {output_dir}')
    if missing:
        for p in missing: print(f'Missing input: {p}')
        if args.check_only:
            return 1
        raise FileNotFoundError('Input datasets must be supplied before execution')
    if args.check_only: return 0
    os.environ['MS_XAI_DATA_DIR'] = str(data_dir)
    os.environ['MS_XAI_OUTPUT_DIR'] = str(output_dir)
    os.environ.setdefault('MPLBACKEND','Agg')
    from IPython.display import display
    namespace = {'__name__':'__main__','__file__':str(root/'run_pipeline.py'),'display':display}
    selected_steps = [step for step in step_files if int(step.name.split('_',1)[0]) <= args.stop_after_step]
    for idx, step in enumerate(selected_steps,1):
        print(f'\nExecuting {idx}/{len(selected_steps)}: {step.name}', flush=True)
        exec(compile(step.read_text(encoding='utf-8'), str(step), 'exec'), namespace, namespace)
    print(f'\nCompleted through step {args.stop_after_step}. Results: {output_dir}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
