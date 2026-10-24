import numpy as np

import params
import lane


def new_tracker():
    return {"left_fit": None, "right_fit": None, "history": [], "bad_frames": 0}


def has_lines(tracker):
    return tracker is not None and tracker["left_fit"] is not None


def reset(tracker):
    tracker["left_fit"] = None
    tracker["right_fit"] = None
    tracker["history"] = []
    tracker["bad_frames"] = 0


def is_consistent(tracker, left_fit, right_fit, shape):
    if not has_lines(tracker):
        return True
    height, width = shape[:2]
    y = height - 1
    max_shift = params.MAX_SHIFT_RATIO * lane.lane_width_px(width)
    left_shift = abs(lane.fit_x(left_fit, y) - lane.fit_x(tracker["left_fit"], y))
    right_shift = abs(lane.fit_x(right_fit, y) - lane.fit_x(tracker["right_fit"], y))
    return left_shift < max_shift and right_shift < max_shift


def update(tracker, left_fit, right_fit, shape):
    accepted = lane.is_valid(left_fit, right_fit, shape) and is_consistent(tracker, left_fit, right_fit, shape)

    if accepted:
        tracker["history"].append((left_fit, right_fit))
        tracker["history"] = tracker["history"][-params.HISTORY_LENGTH:]
        tracker["bad_frames"] = 0
    else:
        tracker["bad_frames"] += 1
        if tracker["bad_frames"] >= params.MAX_BAD_FRAMES:
            reset(tracker)

    if len(tracker["history"]) == 0:
        return left_fit, right_fit, accepted

    tracker["left_fit"] = np.mean([fits[0] for fits in tracker["history"]], axis=0)
    tracker["right_fit"] = np.mean([fits[1] for fits in tracker["history"]], axis=0)
    return tracker["left_fit"], tracker["right_fit"], accepted
