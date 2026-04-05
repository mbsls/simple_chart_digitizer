"""
run_pipeline.py
---------------
Reads charts_config.yaml, runs detect_lines on each image,
saves processed images to figures/, and outputs a single
timeseries.csv with one column per chart filename.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config charts_config.yaml
"""

import sys
import csv
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import detect_lines


# ── Calibration ───────────────────────────────────────────────────────────────

def px_to_date(px, x_min_px, x_max_px, x_min_ms, x_max_ms):
    frac = (px - x_min_px) / (x_max_px - x_min_px)
    ms   = x_min_ms + frac * (x_max_ms - x_min_ms)
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')

def px_to_price(py, y_top_px, y_origin_px, y_max_val, y_min_val):
    frac  = (py - y_top_px) / (y_origin_px - y_top_px)
    return round(y_max_val - frac * (y_max_val - y_min_val), 2)

def extract_series(points, plot, x_min_date, x_max_date, y_min, y_max):
    x_min_ms = datetime.strptime(x_min_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000
    x_max_ms = datetime.strptime(x_max_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000

    date_prices = defaultdict(list)
    for px, py in points:
        date  = px_to_date(px, plot['x_min'], plot['x_max'], x_min_ms, x_max_ms)
        price = px_to_price(py, plot['y_top'], plot['y_origin'], y_max, y_min)
        date_prices[date].append(price)

    rows = [(d, max(ps)) for d, ps in sorted(date_prices.items())]
    rows = [(d, p) for d, p in rows if y_min <= p <= y_max]
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='charts_config.yaml')
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    script_dir = Path(__file__).parent

    charts     = cfg['charts']
    output_cfg = cfg.get('output', {})
    figures_dir = script_dir / output_cfg.get('figures_dir', 'figures')
    csv_name    = output_cfg.get('csv_filename', 'timeseries.csv')

    figures_dir.mkdir(exist_ok=True)

    # images directory: from YAML config, falling back to the config file's folder
    img_dir = Path(cfg.get('images_dir', str(config_path.parent)))

    all_series = {}   # filename -> [(date, price), ...]

    for chart in charts:
        filename   = chart['filename']
        img_path   = img_dir / filename
        print(f"\n{'='*50}")
        print(f"Processing: {filename}")

        result = detect_lines.process(str(img_path))
        plot   = result['plot']

        # get main line points
        if not result['clusters']:
            print(f"  WARNING: no data line detected, skipping")
            continue

        main_info = sorted(result['clusters'].items(),
                           key=lambda c: -len(c[1]['points']))[0][1]
        points = main_info['points']

        series = extract_series(
            points, plot,
            chart['x_min_date'], chart['x_max_date'],
            chart['y_min'],      chart['y_max']
        )
        all_series[filename] = dict(series)
        print(f"  Extracted {len(series)} data points")
        print(f"  Date range: {series[0][0]} → {series[-1][0]}")
        print(f"  Price range: ${min(p for _,p in series):.2f} – ${max(p for _,p in series):.2f}")

        # save processed image
        fig_path = figures_dir / filename
        import shutil
        processed = Path(detect_lines.__file__).parent / (Path(filename).stem + '_processed.png')
        shutil.copy(str(processed), str(fig_path))
        print(f"  Saved figure: {fig_path}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    # Collect all dates across all series
    all_dates = sorted(set(d for series in all_series.values() for d in series))
    filenames  = [c['filename'] for c in charts if c['filename'] in all_series]

    csv_path = script_dir / csv_name
    # Forward-fill missing values: carry the last known price forward
    prev = {fn: '' for fn in filenames}
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date'] + filenames)
        for date in all_dates:
            row = [date]
            for fn in filenames:
                val = all_series[fn].get(date, '')
                if val == '':
                    val = prev[fn]
                prev[fn] = val
                row.append(val)
            writer.writerow(row)

    print(f"\n{'='*50}")
    print(f"CSV saved: {csv_path}  ({len(all_dates)} dates, {len(filenames)} series)")


if __name__ == '__main__':
    main()
