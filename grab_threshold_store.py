"""Persistent, live-settable depth threshold for the YOLO grab gate.

The yolo-ball / yolo-pickplace grab fires when the EWMA-smoothed MiDaS
depth at the ball satisfies D >= threshold (higher = closer). The
threshold defaults to DEFAULT_THRESHOLD and is changed at runtime with
the `set_grab_threshold` IOTCONNECT command; the mode re-reads get()
every frame, so changes apply live with no mode restart. The value
persists to grab_threshold.json; `grab_threshold_reset` deletes the
file and returns to the default.

This intentionally supersedes the taught D_grab/D_stdev gate from
grab_depth.json: the teach flow is still the best way to *discover* a
good value (watch D in the overlay while posing the arm at the grab
point), but the operator sets the gate explicitly.
"""

import json
import os

DEFAULT_THRESHOLD = 750.0
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'grab_threshold.json')

_current = None


def _load():
    global _current
    if os.path.exists(PATH):
        try:
            with open(PATH) as f:
                _current = float(json.load(f)['D_threshold'])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[grab_threshold] load failed ({PATH}): {e} — "
                  f"using default {DEFAULT_THRESHOLD:g}")
            _current = DEFAULT_THRESHOLD
    else:
        _current = DEFAULT_THRESHOLD


def get():
    """Current threshold. Cheap (cached) — safe to call every frame."""
    if _current is None:
        _load()
    return _current


def set_value(value):
    """Validate, persist, and apply a new threshold. Returns the float.

    MiDaS relative depth at this site has been observed roughly in the
    400-1100 range; the wide sanity bound only rejects nonsense input,
    not unusual-but-deliberate values.
    """
    global _current
    v = float(value)
    if not 0 < v < 5000:
        raise ValueError(f"threshold {v:g} outside sane range 1-4999")
    _current = v
    with open(PATH, 'w') as f:
        json.dump({'D_threshold': v}, f, indent=2)
    print(f"[grab_threshold] gate set to D >= {v:g} (saved {PATH})")
    return v


def reset():
    """Back to DEFAULT_THRESHOLD; removes the override file."""
    global _current
    _current = DEFAULT_THRESHOLD
    if os.path.exists(PATH):
        try:
            os.remove(PATH)
            print(f"[grab_threshold] removed {PATH}")
        except OSError as e:
            print(f"[grab_threshold] could not remove {PATH}: {e}")
    return _current
