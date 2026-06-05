# GUI-Owl 1.5 Coordinate Contract

This project distills GUI-Owl 1.5 native Android tool-call behavior into a
Fast-dVLM / block-diffusion student. The student must imitate the teacher's
native output format; it must not learn a new coordinate system.

## Canonical Convention

```text
name: guiowl15_mobile_use_norm1000_xy_v1
teacher: GUI-Owl 1.5
tool: mobile_use
coordinate order: [x, y]
coordinate range: 0..1000
origin: top-left
x=0: left edge
x=1000: right edge
y=0: top edge
y=1000: bottom edge
```

All `target_json` `mobile_use.coordinate` and `mobile_use.coordinate2` values
must follow this convention.

Pixel coordinates are forbidden in training targets. Pixels may only appear
after executor-side conversion immediately before ADB `input tap` or
`input swipe`.

## Valid Tool Calls

```json
{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,932]}}
```

```json
{"name":"mobile_use","arguments":{"action":"swipe","coordinate":[500,850],"coordinate2":[500,250]}}
```

```json
{"name":"mobile_use","arguments":{"action":"system_button","button":"Home"}}
```

```json
{"name":"mobile_use","arguments":{"action":"type","text":"hello"}}
```

## AITW Conversion

AITW raw point columns are normalized as `[y_norm, x_norm]` or split into
`touch_x`, `touch_y`, `lift_x`, and `lift_y`.

Convert to GUI-Owl coordinates with:

```python
coordinate = [round(touch_x * 1000), round(touch_y * 1000)]
coordinate2 = [round(lift_x * 1000), round(lift_y * 1000)]
```

If only pixel coordinates are available:

```python
coordinate = [
    round(pixel_x / image_width * 1000),
    round(pixel_y / image_height * 1000),
]
```

Executor-side conversion to ADB pixels:

```python
pixel_x = round(coordinate[0] / 1000 * screen_width)
pixel_y = round(coordinate[1] / 1000 * screen_height)
```

## Required Dataset Metadata

Every training dataset must provide a machine-readable manifest with:

```json
{
  "coordinate_convention": "guiowl15_mobile_use_norm1000_xy_v1",
  "coordinate_order": "xy",
  "coordinate_range": [0, 1000],
  "target_json_pixel_coordinates": false
}
```

Training code should fail fast when this metadata is missing or when parsed
coordinates fall outside `[0, 1000]`.

## Required Validation

Before training, validate and save:

```text
coord_count
x_min, x_p50, x_p95, x_max
y_min, y_p50, y_p95, y_max
invalid_coord_count
coord_mode counts
action counts
pixel_like_warning
```

If `image_width` and `image_height` exist, a dataset is suspicious when most
coordinates also satisfy `x <= image_width` and `y <= image_height`. That is a
pixel-like pattern and should fail unless the dataset has a documented reason.

## Evaluation Reporting

Report strict and repaired metrics separately:

```text
strict_json_rate
mobile_use_valid_rate
structural_repair_rate
coordinate_rescale_repair_rate, if any
task_success_rate
```

Coordinate rescaling is not structural repair. If an experiment rescales
pixel-looking coordinates at eval time, report it as a separate repair class and
never merge it with strict model correctness.

