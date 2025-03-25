"""Needleman-Wunsch algorithm."""

from typing import List, Tuple


def needleman_wunsch(
    seq1: str,
    seq2: str,
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
    print_matrices: bool = False,
) -> List[Tuple[str, str]]:
    """Find all alignments between two sequences using the Needleman-Wunsch algorithm.

    Args:
        seq1: The first sequence.
        seq2: The second sequence.
        match: Score for a match.
        mismatch: Score for a mismatch.
        gap: Score for a gap.
        print_matrices: Whether to print the score matrix and paths.

    Returns:
        A list of tuples, where each tuple represents an optimal alignment.
    """
    m, n = len(seq1) + 1, len(seq2) + 1
    score_matrix = [[0] * n for _ in range(m)]
    path_matrix = [[[] for _ in range(n)] for _ in range(m)]

    # Initialize the first row and column of the score matrix and path matrix.
    for i in range(m):
        score_matrix[i][0] = i * gap
        if i > 0:
            path_matrix[i][0].append((i - 1, 0))
    for j in range(n):
        score_matrix[0][j] = j * gap
        if j > 0:
            path_matrix[0][j].append((0, j - 1))

    # Fill the score matrix and path matrix.
    for i in range(1, m):
        for j in range(1, n):
            match_score = score_matrix[i - 1][j - 1] + (
                match if seq1[i - 1] == seq2[j - 1] else mismatch
            )
            delete_score = score_matrix[i - 1][j] + gap
            insert_score = score_matrix[i][j - 1] + gap

            score_matrix[i][j] = max(match_score, delete_score, insert_score)

            # Store the paths that lead to the maximum score.
            if score_matrix[i][j] == match_score:
                path_matrix[i][j].append((i - 1, j - 1))
            if score_matrix[i][j] == delete_score:
                path_matrix[i][j].append((i - 1, j))
            if score_matrix[i][j] == insert_score:
                path_matrix[i][j].append((i, j - 1))

    # Print the score matrix and path matrix.
    if print_matrices:
        print("Score matrix:")
        for row in score_matrix:
            print(" ".join(f"{cell:3}" for cell in row))
        print("Path matrix:")
        for row in path_matrix:
            print(row)

    # Backtrack to find all alignments.
    def backtrack(i, j, aligned_seq1, aligned_seq2, alignments):
        if i == 0 and j == 0:
            alignments.append((aligned_seq1, aligned_seq2))
            return

        for prev_i, prev_j in path_matrix[i][j]:
            new_aligned_seq1 = aligned_seq1
            new_aligned_seq2 = aligned_seq2

            if prev_i == i - 1 and prev_j == j - 1:  # Diagonal move
                new_aligned_seq1 = seq1[i - 1] + new_aligned_seq1
                new_aligned_seq2 = seq2[j - 1] + new_aligned_seq2
            elif prev_i == i - 1:  # Vertical move
                new_aligned_seq1 = seq1[i - 1] + new_aligned_seq1
                new_aligned_seq2 = "-" + new_aligned_seq2
            else:  # Horizontal move
                new_aligned_seq1 = "-" + new_aligned_seq1
                new_aligned_seq2 = seq2[j - 1] + new_aligned_seq2

            backtrack(prev_i, prev_j, new_aligned_seq1, new_aligned_seq2, alignments)

    alignments = []
    backtrack(m - 1, n - 1, "", "", alignments)

    return alignments


if __name__ == "__main__":
    seq1 = "GATT"
    seq2 = "GCAT"

    alignments = needleman_wunsch(seq1, seq2, print_matrices=True)

    print("Alignments:")
    for aligned_seq1, aligned_seq2 in alignments:
        print(aligned_seq1)
        print(aligned_seq2)
