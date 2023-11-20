from trackreid.configs.reid_constants import reid_constants


class TrackedObjectFilter:
    """
    The TrackedObjectFilter class is used to filter tracked objects based on their
    confidence and the number of frames they have been observed in.

    Args:
        confidence_threshold (float): The minimum mean confidence level required for a tracked
        object to be considered valid.
        frames_seen_threshold (int): The minimum number of frames a tracked object
        must be observed in to be considered valid.
    """

    def __init__(self, confidence_threshold, frames_seen_threshold):
        self.confidence_threshold = confidence_threshold
        self.frames_seen_threshold = frames_seen_threshold

    def update(self, tracked_object):
        """
        The update method is used to update the state of a tracked object based on its confidence
        and the number of frames it has been observed in.

        If the tracked object's state is TRACKER_OUTPUT, and its mean confidence is greater than the
        confidence_threshold, and it has been observed in more frames than the frames_seen_threshold,
        its state is updated to FILTERED_OUTPUT.

        If the tracked object's mean confidence is less than the confidence_threshold, its state is
        updated to TRACKER_OUTPUT.

        Args:
            tracked_object (TrackedObject): The tracked object to update.
        """
        if tracked_object.get_state() == reid_constants.STATES.TRACKER_OUTPUT:
            if (
                tracked_object.metadata.mean_confidence() > self.confidence_threshold
                and tracked_object.metadata.observations >= self.frames_seen_threshold
            ):
                tracked_object.state = reid_constants.STATES.FILTERED_OUTPUT

        elif tracked_object.metadata.mean_confidence() < self.confidence_threshold:
            tracked_object.state = reid_constants.STATES.TRACKER_OUTPUT
