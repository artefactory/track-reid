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
    def linear_assigment(cost_matrix, candidates, switchers):
        _, _, row_cols = lap.lapjv(
            cost_matrix, extend_cost=True, cost_limit=reid_constants.MATCHES.DISALLOWED_MATCH - 0.1
        )

        matches = []
        for candidate_idx, switcher_idx in enumerate(row_cols):
            if switcher_idx >= 0:
                matches.append({candidates[candidate_idx]: switchers[switcher_idx]})

        return matches
