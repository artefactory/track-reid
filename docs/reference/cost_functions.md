# Cost functions

In the codebase, a cost function is used to measure the dissimilarity between two objects, specifically [TrackedObjects](tracked_object.md) instances. The cost function is a crucial part of the matching process in the [Matcher class](matcher.md). It calculates a cost matrix, where each element represents the cost of assigning a candidate to a switcher.

The cost function affects the behavior of the matching process in the following ways:

1. **Determining Matches**: The cost function is used to determine the best matches between candidates and switchers. The lower the cost, the higher the likelihood that two objects are the same.

2. **Influencing Match Quality**: The choice of cost function can greatly influence the quality of the matches. For example, a cost function that calculates the Euclidean distance between the centers of bounding boxes might be more suitable for tracking objects in a video, while a cost function that calculates the absolute difference between confidence values might be more suitable for matching objects based on their detection confidence.

3. **Setting Match Thresholds**: The cost function also plays a role in setting thresholds for matches. In the [Matcher class](matcher.md), if the cost exceeds a certain threshold, the match is discarded.

You can provide a custom cost function to the reidentification process. For more information, please refer to [this documentation](../custom_cost_selection.md).

:::trackreid.cost_functions
