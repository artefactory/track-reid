# Selection Functions

In the codebase, a selection function is used to determine whether two objects, specifically [TrackedObjects](tracked_object.md) instances, should be considered for matching. The selection function is a key part of the matching process in the [Matcher class](matcher.md).

The selection function influences the behavior of the matching process in the following ways:

1. **Filtering Candidates**: The selection function is used to filter out pairs of objects that should not be considered for matching. This can help reduce the computational complexity of the matching process by reducing the size of the cost matrix.

2. **Customizing Matching Criteria**: The selection function allows you to customize the criteria for considering a pair of objects for matching. For example, you might want to only consider pairs of objects that belong to the same category, or pairs of objects that belong to the same area / zone.

3. **Improving Match Quality**: By carefully choosing or designing a selection function, you can improve the quality of the matches. For example, a selection function that only considers pairs of objects with similar appearance features might lead to more accurate matches.

The selection function should return a boolean value. A return value of `True` or `1` indicates that the pair of objects should be considered for matching, while a return value of `False` or `0` indicates that the pair should not be considered.

You can provide a custom selection function to the reidentification process. For more information, please refer to [this documentation](../custom_cost_selection.md).

:::trackreid.selection_functions
