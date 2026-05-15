import math
from collections import Counter
from typing import Sequence
from numpy.typing import NDArray



def entropy(labels:Sequence[int|str]) -> float:
    if labels is None:
        raise TypeError("labels must not be empty")
    try:
        hash(labels[0])
    except TypeError:
        raise TypeError("labels must be hashable")

    count = Counter(labels)
    total = len(labels)

    ent = 0.0
    for c in count.values():
        probability = c / total
        ent -= probability * math.log2(probability)

    return ent


def information_gain(parent_labels: Sequence[int|str], left_labels: Sequence[int|str], right_labels: Sequence[int|str]) -> float:
    """Calculate the information gain from splitting parent_labels into left_labels and right_labels."""
    if parent_labels is None:
        raise TypeError("parent labels must not be empty")
    if left_labels is None and right_labels is None:
        raise TypeError ("at least one of the child label sets must be non-empty")

    parent_entropy = entropy(parent_labels)
    total = len(parent_labels)

    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total

    weighted_child_entropy = 0.0

    if left_labels:
        weighted_child_entropy += left_weight * entropy(left_labels)

    if right_labels:
        weighted_child_entropy += right_weight * entropy(right_labels)

    return parent_entropy - weighted_child_entropy
