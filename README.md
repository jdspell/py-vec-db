# py-vec-db

## high level goals
- implement a simplistic vector database that can store, search and retrieve vectors

## Implementation
1. Naive Implementation
    - Every vector is compared to all the stored vectors to find the most similar one (KNN)
        - Vectors are stored in in-memory (data structure is a dictionary)
    - Brufe force search is utilized to search all vectors in the database

