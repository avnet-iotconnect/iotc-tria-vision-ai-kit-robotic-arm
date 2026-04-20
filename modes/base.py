"""Strategy interface for vision modes.

Each mode owns its own model loading, per-frame inference, and arm-control
decisions. The outer loop in main.py owns the camera, display window, and
IoTConnect command pump — those are shared across all modes.
"""


class Mode:
    name = "base"

    def setup(self, arm):
        """Called once before the camera loop starts. Load models, init state."""
        return None

    def process_frame(self, frame, arm):
        """Process one BGR frame. Return an annotated BGR image to display, or None to show frame as-is."""
        raise NotImplementedError

    def teardown(self, arm):
        """Called once after the loop exits. Release any mode-owned resources."""
        return None
