from typing import Callable, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from track_reid.tracked_object import TrackedObject


class Matcher:
    def __init__(self, cost_function: Callable, selection_function: Callable) -> None:
        self.cost_function = cost_function
        self.selection_function = selection_function

    def compute_cost_matrix(
        self, objects1: List[TrackedObject], objects2: List[TrackedObject]
    ) -> np.ndarray:
        """Computes a cost matrix of size [M, N] between a list of M TrackedObjects objects1,
        and a list of N TrackedObjects objects2.

        Args:
            objects1 (List[TrackedObject]): list of objects to be matched.
            objects2 (List[TrackedObject]): list of candidates for matches.

        Returns:
            np.ndarray: cost to match each pair of objects.
        """
        if not objects1 or not objects2:
            return np.array([])  # Return an empty array if either list is empty

        # Create matrices with all combinations of objects1 and objects2
        objects1_matrix, objects2_matrix = np.meshgrid(objects1, objects2)

        # Use np.vectorize to apply the scoring function to all combinations
        cost_matrix = np.vectorize(self.cost_function)(objects1_matrix, objects2_matrix)

        return cost_matrix

    def compute_selection_matrix(
        self, objects1: List[TrackedObject], objects2: List[TrackedObject]
    ) -> np.ndarray:
        """Computes a selection matrix of size [M, N] between a list of M TrackedObjects objects1,
        and a list of N TrackedObjects objects2.

        Args:
            objects1 (List[TrackedObject]): list of objects to be matched.
            objects2 (List[TrackedObject]): list of candidates for matches.

        Returns:
            np.ndarray: cost each pair of objects be matched or not ?
        """
        if not objects1 or not objects2:
            return np.array([])  # Return an empty array if either list is empty

        # Create matrices with all combinations of objects1 and objects2
        objects1_matrix, objects2_matrix = np.meshgrid(objects1, objects2)

        # Use np.vectorize to apply the scoring function to all combinations
        selection_matrix = np.vectorize(self.selection_function)(objects1_matrix, objects2_matrix)

        return selection_matrix

    def match(
        self, objects1: List[TrackedObject], objects2: List[TrackedObject]
    ) -> List[Dict[TrackedObject, TrackedObject]]:
        """Computes a dict of matching between objects in list objects1 and objects in objects2.

        Args:
            objects1 (List[TrackedObject]): list of objects to be matched.
            objects2 (List[TrackedObject]): list of candidates for matches.

        Returns:
            List[Dict[TrackedObject, TrackedObject]]: list of pairs of TrackedObjects
            if there is a match.
        """
        if not objects1 or not objects2:
            return []  # Return an empty array if either list is empty

        cost_matrix = self.compute_cost_matrix(objects1, objects2)
        selection_matrix = self.compute_selection_matrix(objects1, objects2)

        # Set a large cost value for elements to be discarded
        cost_matrix[selection_matrix == 0] = 1e3

        # Find the best matches using the linear sum assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=False)

        matches = []
        for row, col in zip(row_indices, col_indices):
            matches.append({objects1[col]: objects2[row]})

        return matches
