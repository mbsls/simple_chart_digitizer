# Simple Chart Digitizer

Very simple code to digitize very simple charts.

Extract time-series data from chart images. Detects colored lines, maps pixel coordinates to dates and values, and outputs a CSV.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Configure

Edit `charts_config.yaml`:

```yaml
images_dir: ./images       # folder containing your chart screenshots

charts:
  - filename: chart1.png
    x_min_date: "2023-01-01"  # date at the left edge of the plot area
    x_max_date: "2026-01-01"  # date at the right edge
    y_min: 0                   # value at the bottom of the plot area
    y_max: 100                 # value at the top

output:
  figures_dir: figures         # where annotated images are saved
  csv_filename: timeseries.csv
```

### 2. Run the pipeline

```bash
python run_pipeline.py --config charts_config.yaml
```

This will:
- Detect the plot area and data lines in each image
- Extract (date, value) pairs
- Save annotated images to `figures/`
- Output `timeseries.csv` with all series

### 3. Plot results

```bash
python plot_timeseries.py
```

Saves `timeseries_plot.png` with all series on one chart.

### Individual image processing

```bash
python detect_lines.py image.png
```

## How it works

1. **Plot area detection** — finds axis lines to determine the data region
2. **Pixel collection** — identifies non-background colored pixels, keeping only the topmost per column
3. **Reference line removal** — filters out flat horizontal lines (e.g. gridlines)
4. **Color clustering** — merges similar colors into distinct data lines
5. **Coordinate mapping** — converts pixel positions to dates and values using the axis calibration from the config
