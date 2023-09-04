import numpy as np

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.similarity_metrics import dot_product


def test_dot_product_calculates_correctly():
    test_vector1 = np.array([1, 2, 3])
    test_vector2 = np.array([4, 5, 6])
    assert np.array_equal(test_vector1.dot(test_vector2), dot_product(test_vector1, test_vector2))

