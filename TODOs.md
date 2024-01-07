# TODOs

## Tecnhiques

- [ ] Min hashing
- [ ] LSH nearest neighbor search
- [x] K-means clustering standard
- [x] Clustroids instead of centroids
- [ ] Recommendation systems
  - [ ] Content-based recommendation systems
- [ ] BFR clustering (normally distributed clusters around centroids in Euclidean space) **PROBABLY NOT**
- [ ] CURE clustering (non-convex clusters in Euclidean space) **PROBABLY NOT**
- [x] DBSCAN clustering (arbitrary shaped clusters in Euclidean space) **MANDATORY**

- Dimensionality reduction (on feature matrix)
  - [ ] SVD


## Distance functions

- [x] Euclidean distance using raw feature vectors
- [ ] Edit distance


## Modifications

- [ ] Use of a different dataset where the divergence is driver-dependent

## k-Shingling

1. [x] Define k
2. [x] Hash the shingles only for cities (3-shingles) ("Rome", "Milano", "Napoli") -> (1, 2, 3) -> Hash -> (92912)
3. [x] add to the set of shingles the merchandise, length = len(merchandise) (qnt-hot encoding)
4. [x] Now we a set that defines completely a route (no trip's length dependent but insensitive to "same difference" merchandise)
5. [ ] Similarity between routes: Jaccard similarity between sets of shingles (where the merchandise is normalized and min max used in formula)
       $ J_q(A, B) = (∑_{x ∈ A ∩ B} min(q(x) in A, q(x) in B)) / (∑_{x ∈ A ∪ B} max(q(x) in A, q(x) in B)) $
6. [ ] Minhashing: 20 hash functions generating permutations then take first index where 1 is found
7. [ ] LSH: 5 bands of 4 rows each, hash each band and then compare the hash values of the bands
8. [ ] Clustering: DBSCAN, k-means, clustroids
