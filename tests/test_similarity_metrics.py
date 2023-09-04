import numpy as np

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.similarity_metrics import dot_product,cosine_similarity


test_vector1 = np.array([1, 2, 3])
test_vector2 = np.array([4, 5, 6])

test_vec_dot_prod = 32
test_vec1_norm = 3.7416573867739413
test_vec2_norm = 8.774964387392123

def test_dot_product_calculates_correctly():
    assert np.array_equal(test_vector1.dot(test_vector2), dot_product(test_vector1, test_vector2))

def test_cosine_similarity_calc_correct():
    test_cosine_similarity = test_vec_dot_prod / (test_vec1_norm * test_vec2_norm)
    assert cosine_similarity(test_vector1, test_vector2) == test_cosine_similarity

