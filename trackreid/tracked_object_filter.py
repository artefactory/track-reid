from trackreid.constants.reid_constants import reid_constants


class TrackedObjectFilter:
    def __init__(self, confidence_threshold, frames_seen_threshold):
        self.confidence_threshold = confidence_threshold
        self.frames_seen_threshold = frames_seen_threshold

    def update(self, tracked_object):
        if tracked_object.get_state() == reid_constants.TRACKER_OUTPUT:
            if (
                tracked_object.metadata.mean_confidence() > self.confidence_threshold
                and tracked_object.metadata.observations >= self.frames_seen_threshold
            ):
                tracked_object.state = reid_constants.FILTERED_OUTPUT

        elif tracked_object.metadata.mean_confidence() < self.confidence_threshold:
            tracked_object.state = reid_constants.TRACKER_OUTPUT
