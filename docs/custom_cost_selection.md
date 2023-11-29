# Designing custom cost and selection functions

## Custom cost function

In our codebase, a cost function is utilized to quantify the dissimilarity between two objects, specifically instances of [TrackedObject](reference/tracked_object.md). The cost function plays a pivotal role in the matching process within the [Matcher class](reference/matcher.md), where it computes a cost matrix. Each element in this matrix represents the cost of assigning a candidate to a switcher. For a deeper understanding of cost functions, please refer to the [related documentation](reference/cost_functions.md).

When initializing the [ReidProcessor](reference/reid_processor.md), you have the option to provide a custom cost function. The requirements for designing one are as follows:

- The cost function must accept 2 [TrackedObjects](reference/tracked_object.md) instances: a candidate (a new object that appears and can potentially be matched), and a switcher (an object that has been lost and can potentially be re-matched).
- All the [metadata](reference/tracked_object_metadata.md) of each [TrackedObject](reference/tracked_object.md) can be utilized to compute a cost.
- If additional metadata is required, you should modify the [metadata](reference/tracked_object_metadata.md) class accordingly. Please refer to the [developer quickstart documentation](quickstart_dev.md) if needed.

Here is an example of an Intersection over Union (IoU) distance function that you can use:

```python
def bounding_box_iou_distance(candidate: TrackedObject, switcher: TrackedObject) -> float:
    """
    Calculates the Intersection over Union (IoU) between the bounding boxes of two TrackedObjects.
    This measure is used as a measure of similarity between the two objects, with a higher IoU
    indicating a higher likelihood of the objects being the same.

    Args:
        candidate (TrackedObject): The first TrackedObject.
        switcher (TrackedObject): The second TrackedObject.

    Returns:
        float: The IoU between the bounding boxes of the two TrackedObjects.
    """
    # Get the bounding boxes from the Metadata of each TrackedObject
    bbox1 = candidate.metadata.bbox
    bbox2 = switcher.metadata.bbox

    # Calculate the intersection of the bounding boxes
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # If the bounding boxes do not overlap, return 0
    if x2 < x1 or y2 < y1:
        return 0.0

    # Calculate the area of the intersection
    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate the area of each bounding box
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return 1 - iou

```

Next, pass this function during the initialization of your [ReidProcessor](reference/reid_processor.md):

```python
reid_processor = ReidProcessor(cost_function_threshold=0.3,
                               cost_function = bounding_box_iou_distance,
                               filter_confidence_threshold=...,
                               filter_time_threshold=...,
                               max_attempt_to_match=...,
                               max_frames_to_rematch=...,
                               save_to_txt=True,
                               file_path="your_file.txt")
```

In this case, candidates and switchers with bounding boxes will be matched if their IoU is below 0.7. Among possible matches, the two bounding boxes with the lowest cost (i.e., larger IoU) will be matched. You can use all the available metadata. For instance, here is an example of a cost function based on the difference in confidence:

```python
def confidence_difference(candidate: TrackedObject, switcher: TrackedObject) -> float:
    """
    Calculates the absolute difference between the confidence values of two TrackedObjects.
    This measure is used as a measure of dissimilarity between the two objects, with a smaller difference
    indicating a higher likelihood of the objects being the same.

    Args:
        candidate (TrackedObject): The first TrackedObject.
        switcher (TrackedObject): The second TrackedObject.

    Returns:
        float: The absolute difference between the confidence values of the two TrackedObjects.
    """
    # Get the confidence values from the Metadata of each TrackedObject
    confidence1 = candidate.metadata.confidence
    confidence2 = switcher.metadata.confidence

    # Calculate the absolute difference between the confidence values
    difference = abs(confidence1 - confidence2)

    return difference

```

Then, pass this function during the initialization of your [ReidProcessor](reference/reid_processor.md):

```python
reid_processor = ReidProcessor(cost_function_threshold=0.1,
                               cost_function = confidence_difference,
                               filter_confidence_threshold=...,
                               filter_time_threshold=...,
                               max_attempt_to_match=...,
                               max_frames_to_rematch=...,
                               save_to_txt=True,
                               file_path="your_file.txt")
```

In this case, candidates and switchers will be matched if their confidence is similar, with a threshold acceptance of 0.1. Among possible matches, the two objects with the lowest cost (i.e., lower confidence difference) will be matched.

## Custom Selection function

In the codebase, a selection function is used to determine whether two objects, specifically [TrackedObjects](reference/tracked_object.md) instances, should be considered for matching. The selection function is a key part of the matching process in the [Matcher class](reference/matcher.md). For a deeper understanding of selection functions, please refer to the [related documentation](reference/selection_functions.md).

Here is an example of a selection function per zone that you can use:

```python

# Define the area of interest, [x_min, y_min, x_max, y_max]
AREA_OF_INTEREST = [0, 0, 500, 500]

def select_by_area(candidate: TrackedObject, switcher: TrackedObject) -> int:

    # Check if both objects are inside the area of interest
    if (candidate.bbox[0] > AREA_OF_INTEREST[0] and candidate.bbox[1] > AREA_OF_INTEREST[1] and
        candidate.bbox[0] + candidate.bbox[2] < AREA_OF_INTEREST[2] and candidate.bbox[1] + candidate.bbox[3] < AREA_OF_INTEREST[3] and
        switcher.bbox[0] > AREA_OF_INTEREST[0] and switcher.bbox[1] > AREA_OF_INTEREST[1] and
        switcher.bbox[0] + switcher.bbox[2] < AREA_OF_INTEREST[2] and switcher.bbox[1] + switcher.bbox[3] < AREA_OF_INTEREST[3]):
        return 1
    else:
        return 0

```

Then, pass this function during the initialization of your [ReidProcessor](reference/reid_processor.md):

```python
reid_processor = ReidProcessor(selection_function = select_by_area,
                               filter_confidence_threshold=...,
                               filter_time_threshold=...,
                               max_attempt_to_match=...,
                               max_frames_to_rematch=...,
                               save_to_txt=True,
                               file_path="your_file.txt")
```

In this case, candidates and switchers will be considerated for matching if they belong to the same zone. You can of course combine selection functions, for instance to selection only switchers and candidates that belong to the same area and belong to the same category.
