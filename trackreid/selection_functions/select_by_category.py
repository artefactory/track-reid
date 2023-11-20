from trackreid.tracked_object import TrackedObject


def select_by_category(candidate: TrackedObject, switcher: TrackedObject) -> int:
    """
    Compares the categories of two TrackedObject instances.
    This selection function is used as a measure of similarity between the two objects,
    matches are discard if this function returns 0.

    Args:
        candidate (TrackedObject): The first TrackedObject instance.
        switcher (TrackedObject): The second TrackedObject instance.

    Returns:
        int: Returns 1 if the categories of the two objects are the same, otherwise returns 0.
    """
    # Compare the categories of the two objects
    return 1 if candidate.category == switcher.category else 0
