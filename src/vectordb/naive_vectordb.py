import numpy as np
from collections import defaultdict
from typing import List, Tuple
from utils import similarity_metrics as sm



class NaiveVectorDB:
    # we can use a dict as the main data structure for storing and retrieving our vectors
    # this database is pretty simple and would likely need a more complex data structure to scale up
    # dict/ hash table will provide fast inserts and lookups which is great for this use case
    def __init__(self):
        self.vectors = defaultdict(np.ndarray)
    
    def insert(self, key: str, vector: np.ndarray) -> None:
        self.vectors[key] = vector

    def retrieve(self, key: str) -> np.ndarray:
        return self.vectors.get(key)
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        # iterate through all vectors in the database and calculate a similatiry score
        similarities = [(key, sm.cosine_similarity(query_vector, vector)) for key, vector in self.vectors.items()]
        # sort the vectors in the database by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        # return top k results
        return similarities[:k]