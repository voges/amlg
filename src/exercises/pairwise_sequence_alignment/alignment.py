"""Module containing functionality for pairwise sequence alignment."""

import random
from typing import Any, Callable, List, Tuple

import numpy as np
import numpy.typing as npt


# pylint: disable=too-many-statements
def make_align(
    mode: str, match: int = 1, mismatch: int = -1, gap: int = -1
) -> Callable:
    """Make and set up the align function.

    This function sets up a couple of free variables that are used by the
    `align()` function as well as some helper functions. This is implementing
    the concept of a closure.

    The `mode` specifies whether to perform global alignment (end-to-end) using the
    Needleman-Wunsch algorithm, or local alignment using the Smith-Waterman algorithm.

    Args:
        mode: The alignment mode. Either "global" or "local".
        match: The score for a match.
        mismatch: The score/penalty for a mismatch.
        gap: The score/penalty for a gap.

    Returns:
        align: The align function.
    """
    # Check if the mode is valid.
    valid_modes = ["global", "local"]
    if mode not in valid_modes:
        raise RuntimeError(f"Invalid mode: {mode}")

    def _init_scoring_mat(a_len: int, b_len: int) -> npt.NDArray[Any]:
        """Initialize the scoring matrix.

        The are two cases here:
        1. Global aligment: Given there is no 'top' or 'top-left' cells for the
           first row (first column) only the existing cell to the left (top)
           can be used to calculate the score of each cell. Hence, as this
           represents a gap, the gap score is added for each shift to the
           right (bottom).
        2. Local alignment: The first row and column are set to zero.

        Args:
            a_len: The length of the first sequence.
            b_len: The length of the second sequence.

        Returns:
            scoring_mat: The initialized scoring matrix.
        """
        scoring_mat = np.zeros(shape=((a_len + 1), (b_len + 1)))
        a_stop = a_len * gap
        b_stop = b_len * gap
        if mode == "local":
            a_stop = 0
            b_stop = 0
        scoring_mat[:, 0] = np.linspace(start=0, stop=a_stop, num=a_len + 1)
        scoring_mat[0, :] = np.linspace(start=0, stop=b_stop, num=b_len + 1)
        return scoring_mat

    def _init_ptr_mat(a_len: int, b_len: int) -> npt.NDArray[Any]:
        """Initialize the pointer matrix.

        Initialize a pointer matrix containing pointers to trace through an
        optimal alignment. The cells which give the highest candidate scores
        must be recorded. Therefore we store pointers to trace through an
        optimal alignment. We use 0 for top-left, 1 for left, and 2 for top.

        Args:
            a_len: The length of the first sequence.
            b_len: The length of the second sequence.

        Returns:
            ptr_mat: The initialized pointer matrix.
        """
        ptr_mat = np.zeros(shape=((a_len + 1), (b_len + 1)))
        ptr_mat[:, 0] = 2
        ptr_mat[0, :] = 1
        return ptr_mat

    def align(
        seq_a: List[str], seq_b: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str], npt.NDArray[Any]]:
        """Perform local or global pairwise sequence alignment.

        The Needleman-Wunsch algorithm finds the optimal global alignment
        (end-to-end) between two sequences. The Smith-Waterman algorithm finds
        the optimal local alignment between two sequences. It is a variation
        of the Needleman-Wunsch algorithm.

        Args:
            seq_a: The first sequence.
            seq_b: The second sequence.

        Returns:
            seq_a_aln: The aligned first sequence. Gaps are filled with "-".
            seq_b_aln: The aligned second sequence. Gaps are filled with "-".
            warp_path_a: The warping path of the first sequence.
            warp_path_b: The warping path of the second sequence.
            scoring_mat: The scoring matrix.
        """
        # Initialize the scoring and pointer matrices.
        scoring_mat = _init_scoring_mat(a_len=len(seq_a), b_len=len(seq_b))
        ptr_mat = _init_ptr_mat(a_len=len(seq_a), b_len=len(seq_b))

        # Fill in the rest of the scoring matrix and construct the pointer
        # matrix. For each cell, we store the pointer of the first maximum
        # that we find. Ideally, all alternative paths should be traced.
        for i, _ in enumerate(seq_a):
            for j, _ in enumerate(seq_b):
                trace = np.zeros(shape=3)
                if seq_a[i] == seq_b[j]:
                    trace[0] = scoring_mat[i, j] + match
                else:
                    trace[0] = scoring_mat[i, j] + mismatch
                trace[1] = scoring_mat[i, (j + 1)] + gap
                trace[2] = scoring_mat[(i + 1), j] + gap
                trace_max = np.max(trace)
                scoring_mat[(i + 1), (j + 1)] = np.max(trace)
                if mode == "local" and trace_max < 0:
                    scoring_mat[(i + 1), (j + 1)] = 0
                else:
                    ptr_mat[(i + 1), (j + 1)] = np.argmax(trace)

        # Set the traceback start.
        i = len(seq_a)
        j = len(seq_b)
        if mode == "local":
            coords = np.argwhere(scoring_mat == np.max(scoring_mat))
            traceback_start = coords[0]
            if len(coords) > 1:
                traceback_start = random.choice(seq=coords)
                print(
                    f"Multiple possible traceback starts: {", ".join(map(str, coords))}"
                )
                print(f"Chose random traceback start from choices: {traceback_start}")
            i = traceback_start[0]
            j = traceback_start[1]

        # Traceback.
        seq_a_aln = []
        seq_b_aln = []
        warp_path_a = []
        warp_path_b = []
        while (mode == "global" and (i > 0 or j > 0)) or (
            mode == "local" and scoring_mat[i, j] > 0
        ):
            warp_path_a.append(i)
            warp_path_b.append(j)
            if ptr_mat[i, j] == 0:  # top-left
                seq_a_aln.append(seq_a[i - 1])
                seq_b_aln.append(seq_b[j - 1])
                i -= 1
                j -= 1
            elif ptr_mat[i, j] == 1:  # left
                seq_a_aln.append(seq_a[i - 1])
                seq_b_aln.append("-")
                i -= 1
            elif ptr_mat[i, j] == 2:  # top
                seq_a_aln.append("-")
                seq_b_aln.append(seq_b[j - 1])
                j -= 1

        # Reverse the sequences.
        seq_a_aln.reverse()
        seq_b_aln.reverse()

        # Reverse the warping paths and let them start with 0.
        warp_path_a.reverse()
        warp_path_b.reverse()
        warp_path_a = [(i - 1) for i in warp_path_a]
        warp_path_b = [(j - 1) for j in warp_path_b]

        # Remove the first row and column from the scoring matrix.
        scoring_mat = scoring_mat[:, 1:]
        scoring_mat = scoring_mat[1:, :]

        return seq_a_aln, seq_b_aln, warp_path_a, warp_path_b, scoring_mat

    return align
