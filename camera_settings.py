"""Persistent camera tuning, applied identically across the demo + calibrators.

Settings live in `camera_settings.json` next to this file. Anything that opens
a `cv2.VideoCapture` (main.py, browser_calibrate.py, browser_calibrate_offset.py)
loads the same file at startup so what you calibrate under is what you demo
under. Cloud commands `camera_setting`, `camera_settings_show`,
`camera_settings_reset` update the file at runtime.

Why this matters: USB cameras default to auto-exposure + auto-white-balance,
which means today's HSV calibration won't match tomorrow's demo. Lock the
camera with `auto_exposure: 1`, `auto_wb: 0`, and a fixed `exposure` value,
then everything stays consistent.

Setting names map to V4L2 / OpenCV properties — see SETTING_PROPS below.
Common starting values (camera-specific; tune for yours):

  {
    "auto_wb":       0,
    "auto_exposure": 1,    # 1 = manual on most V4L2 backends, 3 = auto
    "exposure":      -6,   # range varies; -7 is dark, 0 is bright (Brio/eMeet)
    "saturation":    140
  }
"""

import json
import os

import cv2

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'camera_settings.json')

# Friendly setting name → (cv2 property constant, value caster).
# Cast keeps cloud clients from having to send the right Python type — they
# can send "−7" as a string and we'll float() it.
SETTING_PROPS = {
    'auto_exposure':  (cv2.CAP_PROP_AUTO_EXPOSURE, float),
    'exposure':       (cv2.CAP_PROP_EXPOSURE, float),
    'auto_wb':        (cv2.CAP_PROP_AUTO_WB, float),
    'wb_temperature': (cv2.CAP_PROP_WB_TEMPERATURE, float),
    'brightness':     (cv2.CAP_PROP_BRIGHTNESS, float),
    'contrast':       (cv2.CAP_PROP_CONTRAST, float),
    'saturation':     (cv2.CAP_PROP_SATURATION, float),
    'gain':           (cv2.CAP_PROP_GAIN, float),
    'sharpness':      (cv2.CAP_PROP_SHARPNESS, float),
    'gamma':          (cv2.CAP_PROP_GAMMA, float),
    'hue':            (cv2.CAP_PROP_HUE, float),
}


def load(path=DEFAULT_PATH):
    """Read the settings JSON. Returns {} if the file doesn't exist or is bad."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[camera_settings] load failed ({path}): {e}")
        return {}


def save(settings, path=DEFAULT_PATH):
    try:
        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"[camera_settings] saved to {path}: {settings}")
    except OSError as e:
        print(f"[camera_settings] save failed ({path}): {e}")


def apply(cap, settings):
    """Apply each setting from ``settings`` to an opened cv2.VideoCapture.
    Order matters for some V4L2 stacks: switch auto_* off first, then set the
    fixed values, otherwise the camera reverts. We use a fixed sort order to
    enforce that."""
    if not settings:
        return
    # auto_* keys go first so the corresponding fixed values stick.
    ordered_keys = sorted(settings.keys(),
                          key=lambda k: (0 if k.startswith('auto_') else 1, k))
    for name in ordered_keys:
        value = settings[name]
        prop = SETTING_PROPS.get(name)
        if prop is None:
            print(f"[camera_settings] unknown setting '{name}' — skipping")
            continue
        prop_id, cast = prop
        try:
            ok = cap.set(prop_id, cast(value))
            actual = cap.get(prop_id)
            print(f"[camera_settings] {name}={value} -> set_ok={ok}  read_back={actual}")
        except Exception as e:
            print(f"[camera_settings] failed to set {name}={value}: {e}")


def update_one(name, value, path=DEFAULT_PATH):
    """Load → set one key → save. Returns the updated dict (or None on bad name)."""
    if name not in SETTING_PROPS:
        return None
    settings = load(path)
    settings[name] = value
    save(settings, path)
    return settings


def reset(path=DEFAULT_PATH):
    """Wipe to empty (camera reverts to its V4L2 defaults next open)."""
    save({}, path)
    return {}


def known_setting_names():
    return sorted(SETTING_PROPS.keys())


# ---- runtime re-apply signaling -------------------------------------------
# When a cloud command updates camera_settings.json, the live VideoCapture
# (owned by main.py's _FreshCamera) needs to pick up the new values without
# restarting. The capture loop polls is_dirty() every iteration; when set,
# it reloads the JSON and re-applies it.
_dirty = False


def mark_dirty():
    global _dirty
    _dirty = True


def is_dirty():
    return _dirty


def clear_dirty():
    global _dirty
    _dirty = False
