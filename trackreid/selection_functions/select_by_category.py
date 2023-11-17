from trackreid.tracked_object import TrackedObject


def select_by_category(candidate: TrackedObject, switcher: TrackedObject):
    # Compare the categories of the two objects
    return 1 if candidate.category == switcher.category else 0
