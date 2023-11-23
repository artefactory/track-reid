from typing import Callable, Dict, List, Optional, Union

import lap
import numpy as np

from trackreid.configs.reid_constants import reid_constants
from trackreid.tracked_object import TrackedObject


class Matcher:
    def __init__(
        self,
        cost_function: Callable,
        selection_function: Callable,
        cost_function_threshold: Optional[Union[int, float]] = None,
    ) -> None:
        """
        Initializes the Matcher object with the provided cost function, selection function, and cost function threshold.

        Args:
            cost_function (Callable): A function that calculates the cost of matching two objects. This function should take two TrackedObject instances as input and return a numerical value representing the cost of matching these two objects. A lower cost indicates a higher likelihood of a match.
            selection_function (Callable): A function that determines whether two objects should be considered for matching. This function should take two TrackedObject instances as input and return a binary value (0 or 1). A return value of 1 indicates that the pair should be considered for matching, while a return value of 0 indicates that the pair should not be considered.
            cost_function_threshold (Optional[Union[int, float]]): An optional threshold value for the cost function. If provided, any pair of objects with a matching cost greater than this threshold will not be considered for matching. If not provided, all selected pairs will be considered regardless of their matching cost.

        Returns:
            None
        """  # noqa: E501
        self.cost_function = cost_function
        self.selection_function = selection_function
        self.cost_function_threshold = cost_function_threshold

    def compute_cost_matrix(
        self, candidates: List[TrackedObject], switchers: List[TrackedObject]
    ) -> np.ndarray:
        """Computes a cost matrix of size [M, N] between a list of M TrackedObjects candidates,
        and a list of N TrackedObjects switchers.

        Args:
            candidates (List[TrackedObject]): list of candidates for matches.
            switchers (List[TrackedObject]): list of objects to be matched.

        Returns:
            np.ndarray: cost to match each pair of objects.
        """
        if not candidates or not switchers:
            return np.array([])  # Return an empty array if either list is empty

        # Create matrices with all combinations of candidates and switchers
        candidates_matrix, switchers_matrix = np.meshgrid(candidates, switchers)

        # Use np.vectorize to apply the scoring function to all combinations
        cost_matrix = np.vectorize(self.cost_function)(candidates_matrix, switchers_matrix)

        return cost_matrix

    def compute_selection_matrix(
        self, candidates: List[TrackedObject], switchers: List[TrackedObject]
    ) -> np.ndarray:
        """Computes a selection matrix of size [M, N] between a list of M TrackedObjects candidates,
        and a list of N TrackedObjects switchers.

        Args:
            candidates (List[TrackedObject]): list of candidates for matches.
            switchers (List[TrackedObject]): list of objects to be rematched.

        Returns:
            np.ndarray: cost each pair of objects be matched or not ?
        """
        if not candidates or not switchers:
            return np.array([])  # Return an empty array if either list is empty

        # Create matrices with all combinations of candidates and switchers
        candidates_matrix, switchers_matrix = np.meshgrid(candidates, switchers)

        # Use np.vectorize to apply the scoring function to all combinations
        selection_matrix = np.vectorize(self.selection_function)(
            candidates_matrix, switchers_matrix
        )

        return selection_matrix

    def match(
        self, candidates: List[TrackedObject], switchers: List[TrackedObject]
    ) -> List[Dict[TrackedObject, TrackedObject]]:
        """Computes a dict of matching between objects in list candidates and objects in switchers.

        Args:
            candidates (List[TrackedObject]): list of candidates for matches.
            switchers (List[TrackedObject]): list of objects to be matched.

        Returns:
            List[Dict[TrackedObject, TrackedObject]]: list of pairs of TrackedObjects
            if there is a match.
        """
        if not candidates or not switchers:
            return []  # Return an empty array if either list is empty

        cost_matrix = self.compute_cost_matrix(candidates, switchers)
        selection_matrix = self.compute_selection_matrix(candidates, switchers)

        # Set a elements values to be discard at DISALLOWED_MATCH value, large cost
        cost_matrix[selection_matrix == 0] = reid_constants.MATCHES.DISALLOWED_MATCH
        if self.cost_function_threshold is not None:
            cost_matrix[
                cost_matrix > self.cost_function_threshold
            ] = reid_constants.MATCHES.DISALLOWED_MATCH

        matches = self.linear_assigment(cost_matrix, candidates=candidates, switchers=switchers)

        return matches

    @staticmethod
    def linear_assigment(
        cost_matrix: np.ndarray, candidates: List[TrackedObject], switchers: List[TrackedObject]
    ) -> List[Dict[TrackedObject, TrackedObject]]:
        """
        Performs linear assignment on the cost matrix to find the optimal match between candidates and switchers.

        The function uses the Jonker-Volgenant algorithm to solve the linear assignment problem. The algorithm finds the
        optimal assignment (minimum total cost) for the given cost matrix. The cost matrix is a 2D numpy array where
        each cell represents the cost of assigning a candidate to a switcher.

        Args:
            cost_matrix (np.ndarray): A 2D array representing the cost of assigning each candidate to each switcher.
            candidates (List[TrackedObject]): A list of candidate TrackedObjects for matching.
            switchers (List[TrackedObject]): A list of switcher TrackedObjects to be matched.

        Returns:
            List[Dict[TrackedObject, TrackedObject]]: A list of dictionaries where each dictionary represents a match.
            The key is a candidate and the value is the corresponding switcher.
        """
        _, _, row_cols = lap.lapjv(
            cost_matrix, extend_cost=True, cost_limit=reid_constants.MATCHES.DISALLOWED_MATCH - 0.1
        )

        matches = []
        for candidate_idx, switcher_idx in enumerate(row_cols):
            if switcher_idx >= 0:
                matches.append({candidates[candidate_idx]: switchers[switcher_idx]})

        return matches
