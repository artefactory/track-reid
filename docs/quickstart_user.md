# Using the ReidProcessor

The `ReidProcessor` is the entry point of the `track-reid` library. It is used to process and reconcile tracking data, ensuring consistent and accurate tracking of objects over time. Here's a step-by-step guide on how to use it:

## Step 1: Understand the Usage

The reidentification process is applied to tracking results, which are derived from the application of a tracking algorithm on detection results for successive frames of a video. This reidentification process is applied iteratively on each tracking result, updating its internal states during the process.

The `ReidProcessor` needs to be updated with the tracking results for each frame of your
sequence or video. This is done by calling the `update` method that takes 2 arguments:

- `frame_id`: an integer specifying the current frame of the video
- `tracker_output`: a numpy array containing the tracking results for the current frame

## Step 2: Understand the Data Format Requirements

The `ReidProcessor` update function requires a numpy array of tracking results for the current frame as input. This data must meet specific criteria regarding data type and structure.

All input data must be numeric, either integers or floats.
Here's an example of the expected input data format based on the schema:

| bbox (0-3)      | object_id (4) | category (5) | confidence (6) |
|-----------------|---------------|--------------|----------------|
| 50, 60, 120, 80 |       1       |       1      |       0.91     |
| 50, 60, 120, 80 |       2       |       0      |       0.54     |

Each row corresponds to a tracked object.

- The first four columns denote the **bounding box coordinates** in the format (x, y, width, height),
where x and y are the top left coordinates of the bounding box. These coordinates can be either normalized or in pixel units.
These values remain unchanged during the reidentification process.
- The fifth column is the **object ID** assigned by the tracker, which may be adjusted during the reidentification process.
- The sixth column indicates the **category** of the detected object, which may also be adjusted during the reidentification process.
- The seventh column is the confidence score of the detection, which is not modified by the reidentification process.

For additional information, you can utilize `ReidProcessor.print_input_data_requirements()`.

Here's a reformatted example of how the output data should appear, based on the schema:

| frame_id (0) | object_id (1) | category (2) | bbox (3-6)      | confidence (7) | mean_confidence (8) | tracker_id (9) |
|--------------|---------------|--------------|-----------------|----------------|---------------------|----------------|
| 1            | 1             | 1            | 50, 60, 120, 80 | 0.91           | 0.85                | 1              |
| 2            | 2             | 0            | 50, 60, 120, 80 | 0.54           | 0.60                | 2              |

- The first column represents the **frame identifier**, indicating the frame for which the result is applicable.
- The second column is the **object ID** assigned by the reidentification process.
- The third column is the **category** of the detected object, which may be adjusted during the reidentification process.
- The next four columns represent the **bounding box coordinates**, which remain unchanged from the input data.
- The seventh column is the **confidence** of the object, which also remains unchanged from the input data.
- The eighth column indicates the **average confidence** of the detected object over its lifetime, from the beginning of the tracking to the current frame.
- The final column is the **object ID assigned by the tracking algorithm**, before the reidentification process.

You can use `ReidProcessor.print_output_data_format_information()` for more insight.

## Step 3: Understand Necessary Modules

To make ReidProcessor work, several modules are necessary:

- `TrackedObject`: This class represents a tracked object. It is used within the Matcher and ReidProcessor classes.
- `TrackedObjectMetadata`: This class is attached to a tracked object and represents informations and properties about the object.
- `TrackedObjectFilter`: This class is used to filter tracked objects based on certain criteria. It is used within the ReidProcessor class.
- `Matcher`: This class is used to match tracked objects based on a cost function and a selection function. It is initialized within the ReidProcessor class.

The cost and selection functions are key components of the ReidProcessor, as they will drive the matching process between lost objects and new objects during the video. Those two functions are fully customizable and can be passed as arguments of the ReidProcessor at initialization. They both take 2 `TrackedObjects` as inputs, and perform computation based on their metadatas.

- **cost function**: This function calculates the cost of matching two objects. It takes two TrackedObject instances as input and returns a numerical value representing the cost of matching these two objects. A lower cost indicates a higher likelihood of a match. The default cost function is `bounding_box_distance`.

- **selection_function**: This function determines whether two objects should be considered for matching. It takes two TrackedObject instances as input and returns a binary value (0 or 1). A return value of 1 indicates that the pair should be considered for matching, while a return value of 0 indicates that the pair should not be considered. The default selection function is `select_by_category`.

In summary, prior to the matching process, filtering on which objects should be considerated is applied thought the `TrackedObjectFilter`. All objects are represented by the `TrackedObject` class, with its attached metadata represented by `TrackedObjectMetadata`. The `ReidProcessor` then uses the `Matcher` class with a cost function and selection function to match objects.

## Step 4: Initialize ReidProcessor

If you do not want to provide custom cost and selection function, here is an example of ReidProcessor initialization:

```python
reid_processor = ReidProcessor(filter_confidence_threshold=0.1,
                               filter_time_threshold=5,
                               cost_function_threshold=5000,
                               max_attempt_to_match=5,
                               max_frames_to_rematch=500,
                               save_to_txt=True,
                               file_path="your_file.txt")
```

Here is a brief explanation of each argument in the ReidProcessor function, and how you can monitor the `Matcher` and the `TrackedObjectFilter` behaviours:

- `filter_confidence_threshold`: Float value that sets the **minimum average confidence level** for a tracked object to be considered valid. Tracked objects with average confidence levels below this threshold will be ignored.

- `filter_time_threshold`: Integer that sets the **minimum number of frames** a tracked object must be seen with the same id to be considered valid. Tracked objects seen less frames that this threshold will be ignored.

- `cost_function_threshold`: This is a float value that sets the **maximum cost for a match** between a detection and a track. If the cost of matching a detection to a track exceeds this threshold, the match will not be made. Set to None for no limitation.

- `max_attempt_to_match`: This is an integer that sets the **maximum number of attempts to match a tracked object never seen before** to a lost tracked object. If this tracked object never seen before can't be matched within this number of attempts, it will be considered a new stable tracked object.

- `max_frames_to_rematch`: This is an integer that sets the **maximum number of frames to try to rematch a tracked object that has been lost**. If a lost object can't be rematch within this number of frames, it will be considered as lost forever.

- `save_to_txt`: This is a boolean value that determines whether the tracking results should be saved to a text file. If set to True, the results will be saved to a text file.

- `file_path`: This is a string that specifies the path to the text file where the tracking results will be saved. This argument is only relevant if save_to_txt is set to True.

For more information on how to design custom cost and selection functions, refer to [this guide](custom_cost_selection.md).

## Step 5: Run reidentifiaction process

Lets say you have a `dataset` iterable object, composed for each iteartion of a frame id and its associated tracking results. You can call the `ReidProcessor` update class using the following:

```python
for frame_id, tracker_output in dataset:
    corrected_results = reid_processor.update(frame_id = frame_id,                  tracker_output=tracker_output)
```

At the end of the for loop, information about the correction can be retrieved using the `ReidProcessor` properties. For instance, the list of tracked object can be accessed using:

```python
reid_processor.seen_objects()
```
