from numba import njit, prange, jit
from numba_progress import ProgressBar
from numba.core.errors import NumbaWarning

import warnings

import numpy as np

import random
from tqdm import tqdm

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


warnings.simplefilter('ignore', category=NumbaWarning)

def hash_function_hash_code(num_of_hashes,n_col,next_prime):
    coeffA = np.array(random.sample(range(0,n_col*100),num_of_hashes)).reshape((num_of_hashes,1))
    coeffB = np.array(random.sample(range(0,n_col*100),num_of_hashes)).reshape((num_of_hashes,1))

    x = np.arange(n_col).reshape((1,n_col))

    hash_code = (np.matmul(coeffA,x) + coeffB) % next_prime # (num_of_hashes,n_col) so how each column index is permuted

    return hash_code


def minhash_matrices(matrix1,matrix2,num_of_hashes):
    (n_row, n_col) = matrix1.shape
    next_prime = n_col
    hash_code = hash_function_hash_code(num_of_hashes,n_col,next_prime)

    signature1_array = np.empty(shape = (n_row,num_of_hashes))
    signature2_array = np.empty(shape = (matrix2.shape[0],num_of_hashes))

    for row in tqdm(range(n_row), desc="Route matrix"):
        ones_index = np.where(matrix1[row,:]==1)[0]
        signature1_array[row,:] = np.zeros((1,num_of_hashes))
        corresponding_hashes = hash_code[:,ones_index]
        row_signature = np.amin(corresponding_hashes,axis=1).reshape((1,num_of_hashes))

        signature1_array[row,:] = row_signature

    for row in tqdm(range(matrix2.shape[0]), desc="Standard matrix"):
        ones_index = np.where(matrix2[row,:]==1)[0]
        signature2_array[row,:] = np.zeros((1,num_of_hashes))
        corresponding_hashes = hash_code[:,ones_index]
        row_signature = np.amin(corresponding_hashes,axis=1).reshape((1,num_of_hashes))

        signature2_array[row,:] = row_signature

    return signature1_array, signature2_array

def find_band_and_row_values(columns, threshold):
    previous_b = 1
    previous_r = columns
    for b in range(1, columns + 1):
        if columns % b == 0:
            r = columns // b
            if (1 / b) ** (1 / r)  <= threshold:
                if np.abs((1 / previous_b) ** (1 / previous_r) - threshold) < np.abs((1 / b) ** (1 / r) - threshold):
                    return previous_b, previous_r
                return b, r
            previous_b = b
            previous_r = r
    return columns, 1

def lsh(minhash_matrix, thresh_user=0.2):
    # Initialize the signature matrix
    columns = minhash_matrix.shape[1]
    
    # Generate the hash functions
   # hash_functions = [lambda x, a=a, b=b: (a * x + b) % minhash_matrix.shape[1] for a, b in zip(random.sample(range(1000), bands), random.sample(range(1000), bands))]
    hash_function = lambda x: hash(",".join([str(x[i]) for i in range(len(x))]))
    
    # b = bands
    # r = columns//bands
    b, r = find_band_and_row_values(columns, thresh_user)
    # If columns is not divisible by bands
    if columns % b != 0:
        # Find the closest number that makes it divisible
        while columns % b != 0:
            b -= 1
        r = columns // b
        
    print("Bands used in LSH:", b)
    signature_matrix = np.full((minhash_matrix.shape[0], b), np.inf)
    
    # if threshold is 0.8,
    threshold = (1 / b) ** (1 / r) 
    print("Closest LSH threshold:", threshold)
    
    # For each band
    print("Computing hash values of bands...")
    hash_values = np.apply_along_axis(lambda x: hash_function(x) % minhash_matrix.shape[0], 1, minhash_matrix.reshape(-1, r))
    # Reshape the hash values to match the signature matrix
    hash_values = hash_values.reshape(minhash_matrix.shape[0], b)
    # Update the signature matrix
    signature_matrix = hash_values
            
    # find candidate pairs
    print("Finding candidate pairs...")
    candidate_pairs = []
    for i in tqdm(range(signature_matrix.shape[0])):
        # Compute the similarity of the current row with all following rows
        similarities = np.sum(signature_matrix[i+1:, :] == signature_matrix[i, :], axis=1) / b
        # Find the indices of the rows that have a similarity greater than or equal to the threshold
        indices = np.nonzero(similarities >= threshold)[0]
        # Add the pairs to the candidate pairs
        candidate_pairs.extend((i, i+1+index) for index in indices)
    
    return np.array(candidate_pairs)

def lsh_two_matrices(minhash_matrix1, minhash_matrix2, thresh_user=0.2):
    # Initialize the signature matrix
    columns = minhash_matrix1.shape[1]
    
    # Generate the hash function
    def hash_function(x):
        var = hash(",".join([str(x[i]) for i in range(len(x))]))
        return var % minhash_matrix1.shape[0]


    # b = bands
    # r = columns//bands
    b, r = find_band_and_row_values(columns, thresh_user)
    # If columns is not divisible by bands
    if columns % b != 0:
        # Find the closest number that makes it divisible
        while columns % b != 0:
            b -= 1
        r = columns // b
        
    print("Bands used in LSH:", b)
    signature_matrix1 = np.full((minhash_matrix1.shape[0], b), np.inf)
    signature_matrix2 = np.full((minhash_matrix2.shape[0], b), np.inf)
    

    threshold = (1 / b) ** (1 / r) 
    print("Closest LSH threshold:", threshold)

    # For each band
    print("Computing hash values of bands...")
    hash_values1 = np.apply_along_axis(hash_function, 1, minhash_matrix1.reshape(-1, r))
    hash_values2 = np.apply_along_axis(hash_function, 1, minhash_matrix2.reshape(-1, r))


    # Reshape the hash values to match the signature matrix
    hash_values1 = hash_values1.reshape(minhash_matrix1.shape[0], b)
    hash_values2 = hash_values2.reshape(minhash_matrix2.shape[0], b)
    # Update the signature matrix
    signature_matrix1 = hash_values1
    signature_matrix2 = hash_values2
    
    
    # find candidate pairs
    print("Finding candidate pairs...")

    data=[]
    rows=[]
    cols=[]

    for i in tqdm(range(signature_matrix1.shape[0])):
        # Compute the similarity of the current row with all following rows
        similarities = np.sum(signature_matrix2 == signature_matrix1[i, :], axis=1) / b
        # Find the indices of the rows that have a similarity greater than or equal to the threshold
        indices = np.nonzero(similarities >= threshold)[0]

        data.extend(similarities[indices])
        rows.extend([i]*len(indices))
        cols.extend(indices)

    print("Creating sparse LSH matrix...")
    similarity_matrix = coo_matrix((data, (rows, cols)), shape=(minhash_matrix1.shape[0], minhash_matrix2.shape[0])).tocsr()

    return similarity_matrix

@jit(cache=True)
def compute_distance_pairs_merch(sim_matrix, matrix1, matrix1Merch, matrix2, matrix2Merch, progress_proxy, metric="jaccard", alpha=0.8, fusion="mean"):
    n = sim_matrix.shape[0]
    m = sim_matrix.shape[1]
    squareMatrix = np.full((matrix1Merch.shape[0], matrix2Merch.shape[1]), 2)

    for i in prange(n):
        subset1 = matrix1[i].reshape(1, -1)
        subset2 = matrix2[sim_matrix[i].nonzero()[1]]
        min_matrix = np.minimum(subset1, subset2)
        sum_min_matrix = np.sum(min_matrix, axis=-1)
        
        max_matrix = np.maximum(subset1, subset2)
        sum_max_matrix = np.sum(max_matrix, axis=-1)

        route_distance = (np.divide(sum_min_matrix, sum_max_matrix))

        subset1Merch = matrix1Merch[i].reshape(1, -1)
        subset2Merch = matrix2Merch[sim_matrix[i].nonzero()[1]]
        
        if metric == "cosine":
            # COSINE
            normsSubset2Merch = np.sqrt(np.sum(np.power(subset2Merch, squareMatrix), axis=1))
            merch_distance = (((subset1Merch * subset2Merch).sum(axis=1) / (np.sqrt(np.sum(np.power(subset1Merch, squareMatrix),axis=1)) * normsSubset2Merch)) + 1) / 2
        elif metric == "jaccard":
            # JACCARD
            min_matrixMerch = np.minimum(subset1Merch, subset2Merch)
            sum_min_matrixMerch = np.sum(min_matrixMerch, axis=-1)

            max_matrixMerch = np.maximum(subset1Merch, subset2Merch)
            sum_max_matrixMerch = np.sum(max_matrixMerch, axis=-1)

            merch_distance = (np.divide(sum_min_matrixMerch, sum_max_matrixMerch))
            #print("merch_distance", merch_distance.shape)
        else:
            # L2
            merch_distance = 1 / (1 + np.sqrt(np.sum(np.square(subset1Merch - subset2Merch), axis=-1)))


        if fusion == "mean":
            # mean
            sim_matrix[i,sim_matrix[i].nonzero()[1]] = (alpha) * route_distance + (1-alpha) * merch_distance
        elif fusion == "product":
            # product
            sim_matrix[i,sim_matrix[i].nonzero()[1]] = route_distance * merch_distance
        else:
            # weighted product
            weightsRoutes = np.full(sim_matrix[i].nonzero()[1].shape[0], alpha)
            weightsMerch = np.full(sim_matrix[i].nonzero()[1].shape[0], 1-alpha)
            sim_matrix[i,sim_matrix[i].nonzero()[1]] = np.power(route_distance, weightsRoutes) * np.power(merch_distance, weightsMerch)
        
        progress_proxy.update(1)
    
    return sim_matrix


def similarity_minhash_lsh_two_matrices_and_merch(matrix1, matrix1Merch, matrix2, matrix2Merch, thresh_user=0.2, metric="jaccard", alpha=0.8, fusion="mean"):
    
    similarity_matrix = lsh_two_matrices(matrix1,matrix2, thresh_user=thresh_user)
    print("Computing distance  on subset matrix...")
    with ProgressBar(total=matrix1.shape[0]) as progress:
        similarity_matrix = compute_distance_pairs_merch(similarity_matrix, matrix1, matrix1Merch, matrix2, matrix2Merch, progress, metric=metric, alpha=alpha, fusion=fusion)
        
    return similarity_matrix
        


@njit(cache=True, nogil=True, parallel=True)
def compute_subset_similarity_matrix_and_merch(matrix, matrixMerch, progress_proxy, metric="jaccard", alpha=0.8, fusion="mean"):
    n = matrix.shape[0]
    n1 = matrix.shape[1]
    m = matrixMerch.shape[1]
    similarity_pairs = np.zeros((n,n))
    subset2 = matrix
    subset2Merch = matrixMerch
    squareMatrix = np.full((n, m), 2)
    routeWeights = np.full(n, alpha)
    merchWeights = np.full(n, 1-alpha)
    normsSubset2Merch = np.sqrt(np.sum(np.power(subset2Merch, squareMatrix), axis=1))
    for i in prange(n):
        subset1 = matrix[i].reshape(1, -1)
        subset1Merch = matrixMerch[i].reshape(1, -1)
        
        min_matrix = np.minimum(subset1, subset2)
        sum_min_matrix = np.sum(min_matrix, axis=-1)
        
        max_matrix = np.maximum(subset1, subset2)
        sum_max_matrix = np.sum(max_matrix, axis=-1)
        
        if metric == "cosine":
            # COSINE
            distMerch = 1 - (((subset1Merch * subset2Merch).sum(axis=1) / (np.sqrt(np.sum(np.power(subset1Merch, squareMatrix),axis=1)) * normsSubset2Merch)) + 1) / 2
        elif metric == "jaccard":
            # JACCARD
            min_matrix_merch = np.minimum(subset1Merch, subset2Merch)
            sum_min_matrix_merch = np.sum(min_matrix_merch, axis=-1)
            max_matrixMerch = np.maximum(subset1Merch, subset2Merch)
            sum_max_matrixMerch = np.sum(max_matrixMerch, axis=-1)
            distMerch = 1 - (sum_min_matrix_merch / sum_max_matrixMerch)
        else:
            # L2
            distMerch = np.sqrt(np.sum(np.power(subset1Merch - subset2Merch, squareMatrix), axis=1))
        
        
        
        routeDistance = 1 - (sum_min_matrix / sum_max_matrix)
        
        if fusion == "mean":
            # MEAN
            similarity_pairs[i] = (alpha) * routeDistance + (1-alpha) * distMerch
        elif fusion == "product":
            # PRODUCT
            similarity_pairs[i] = routeDistance * distMerch
        else:
            # WEIGHTED PRODUCT
            similarity_pairs[i] = np.power(routeDistance, routeWeights) * np.power(distMerch, merchWeights)
        
        progress_proxy.update(1)
    return similarity_pairs


def jaccard_similarity_minhash_lsh_route_merch(matrix, matrixMerch, thresh_user=0.2, metric="jaccard", alpha=0.8, fusion="mean"):
    pairs = lsh(matrix, thresh_user=thresh_user)
    uniqueRowsSet = set([i for i, j in pairs] + [j for i, j in pairs]) # (1,2) (1,4) (1,5)
    neverSeen = set([i for i in range(matrix.shape[0])]) - uniqueRowsSet
    print("Subset of rows to check similarity:", len(uniqueRowsSet))
    print(" num of pairs", len(uniqueRowsSet)* (len(uniqueRowsSet)-1)/2)
    print(" instead of", matrix.shape[0]*(matrix.shape[0]-1)/2)
    print("improved by", (1 - len(uniqueRowsSet)* (len(uniqueRowsSet)-1)/2 / (matrix.shape[0]*(matrix.shape[0]-1)/2)) *100, "%")
    
        
    print("Computing jaccard similarity on subset matrix...")
    
    sortedUniqueRowsSet = sorted(list(uniqueRowsSet))
    subset_matrix = matrix[sortedUniqueRowsSet]
    subset_matrixMerch = matrixMerch[sortedUniqueRowsSet]
    with ProgressBar(total=len(sortedUniqueRowsSet)) as progress:
        subset_sim_matrix = compute_subset_similarity_matrix_and_merch(subset_matrix, subset_matrixMerch, progress, metric=metric, alpha=alpha, fusion=fusion)
    
    print("Creating mapping...")
    
    # remove never seen rows and map indices
    map_indices = {}
    sortedNeverSeen = sorted(list(neverSeen))
    counter = 0
    for i in range(matrix.shape[0]):
        if i in sortedNeverSeen:
            continue
        map_indices[i] = counter
        counter += 1
        
    map_indices_back = {v: k for k, v in map_indices.items()}
    
    print("Creating sparse similarity matrix...")
    subset_sim_matrix = csr_matrix(subset_sim_matrix, dtype=np.float32)
    
    return subset_sim_matrix, map_indices, map_indices_back

def similarity_two_matrices_and_merch(matrix1,matrix1Merch, matrix2, matrix2Merch, alpha=0.8, fusion="mean"):
    if 1.0 - np.count_nonzero(matrix1) / matrix1.size > 0.5 and 1.0 - np.count_nonzero(matrix2) / matrix2.size > 0.5:
        print("matrix jaccard is sparse")
        
        matrixCSR = csr_matrix(matrix1)
        matrix2CSR = csr_matrix(matrix2)
        
        intersection = np.dot(matrixCSR, matrix2CSR.T)
        #intersection = intersection.todense()
        row_sums = matrix1.sum(axis=1)
        row_sums2 = matrix2.sum(axis=1)
        union = row_sums[:, None] + row_sums2 - intersection
        union = np.where(union == 0, 1, union)
        routeSim = (intersection / union).toarray()
    else:
        intersection = np.dot(matrix1, matrix2.T)
        row_sums = matrix1.sum(axis=1)
        row_sums2 = matrix2.sum(axis=1)
        union = row_sums[:, None] + row_sums2 - intersection
        union = np.where(union == 0, 1, union)
        routeSim = intersection / union
    

    merchSim = cosine_similarity(matrix1Merch, matrix2Merch) #((1 + np.sum(matrix1Merch * matrix2Merch2, axis=-1) / np.linalg.norm(matrix1Merch, axis=-1) / np.linalg.norm(matrix2Merch2, axis=-1)) / 2)
        
        
    if fusion == "mean":
        # MEAN
        return (alpha) * routeSim + (1-alpha) * merchSim
    elif fusion == "product":
        # PRODUCT
        return routeSim * merchSim
    else:
        # WEIGHTED PRODUCT
        return np.power(routeSim, alpha) * np.power(merchSim, 1-alpha)

def similarity_minhash_two_matrices_and_merch(matrix1, matrix1Merch, matrix2, matrix2Merch, metric="jaccard", alpha=0.8, fusion="mean"):
    print("Computing jaccard similarity on full matrices...")
    matrixRoute1 = matrix1[:, None, :]
    matrixRoute2 = matrix2[None, :, :]
    
    min_matrix = np.minimum(matrixRoute1, matrixRoute2)
    sum_min_matrix = np.sum(min_matrix, axis=-1)
    
    max_matrix = np.maximum(matrixRoute1, matrixRoute2)
    sum_max_matrix = np.sum(max_matrix, axis=-1)
    
    routeSim =  (sum_min_matrix / sum_max_matrix)
    
    matrixMerch1 = matrix1Merch[:, None, :]
    matrixMerch2 = matrix2Merch[None, :, :]
    
    if metric == "cosine":
        # COSINE
        merchSim = ((1 + np.sum(matrixMerch1 * matrixMerch2, axis=-1) / np.linalg.norm(matrixMerch1, axis=-1) / np.linalg.norm(matrixMerch2, axis=-1)) / 2)
    elif metric == "jaccard":
        # JACCARD
        min_matrixMerch = np.minimum(matrixMerch1, matrixMerch2)
        sum_min_matrixMerch = np.sum(min_matrixMerch, axis=-1)
        
        max_matrixMerch = np.maximum(matrixMerch1, matrixMerch2)
        sum_max_matrixMerch = np.sum(max_matrixMerch, axis=-1)
        
        merchSim = (sum_min_matrixMerch / sum_max_matrixMerch)
    else:
        # L2
        merchSim = 1 / (1 + np.linalg.norm(matrixMerch1 - matrixMerch2, axis=-1))
        
        
    if fusion == "mean":
        # MEAN
        return (alpha) * routeSim + (1-alpha) * merchSim
    elif fusion == "product":
        # PRODUCT
        return routeSim * merchSim
    else:
        # WEIGHTED PRODUCT
        return np.power(routeSim, alpha) * np.power(merchSim, 1-alpha)

def jaccard_distance_route_merch(matrix, matrixMerch, alpha=0.8, fusion="mean"):
    if 1.0 - np.count_nonzero(matrix) / matrix.size > 0.5:
        print("matrix jaccard is sparse")
        matrixCSR = csr_matrix(matrix)
        
        intersection = np.dot(matrixCSR, matrixCSR.T)
        #intersection = intersection.todense()
        row_sums = matrix.sum(axis=1)
        union = row_sums[:, None] + row_sums - intersection
        union = np.where(union == 0, 1, union)  # avoid division by zero
        routeDist = 1 - (intersection / union).toarray()
    else:
        intersection = np.dot(matrix, matrix.T)
        row_sums = matrix.sum(axis=1)
        union = row_sums[:, None] + row_sums - intersection
        union = np.where(union == 0, 1, union)
        routeDist = 1 - (intersection / union)
    
    
    merchDist = 1 - cosine_similarity(matrixMerch) #((1 + np.sum(matrixMerch * matrixMerch, axis=-1) / np.linalg.norm(matrixMerch, axis=-1) / np.linalg.norm(matrixMerch, axis=-1)) / 2)
        
        
    if fusion == "mean":
        # MEAN
        return (alpha) * routeDist + (1-alpha) * merchDist
    elif fusion == "product":
        # PRODUCT
        return routeDist * merchDist
    else:
        # WEIGHTED PRODUCT
        return np.power(routeDist, alpha) * np.power(merchDist, 1-alpha)

def jaccard_distance_minhash_route_merch(matrix, matrixMerch, metric="jaccard", alpha=0.8, fusion="mean"):
    print("Computing jaccard distance on full matrix...")

    n = matrix.shape[0]
    n1 = matrix.shape[1]
    m = matrixMerch.shape[1]
    
    # compute jaccard similarity route
    
    matrixRoute1 = matrix[:, None, :]
    matrixRoute2 = matrix[None, :, :]
    min_matrix = np.minimum(matrixRoute1, matrixRoute2)
    sum_min_matrix = np.sum(min_matrix, axis=-1)
    
    max_matrix = np.maximum(matrixRoute1, matrixRoute2)
    sum_max_matrix = np.sum(max_matrix, axis=-1)
    
    routeDistance = 1 - (sum_min_matrix / sum_max_matrix)
    
    matrixMerch1 = matrixMerch[:, None, :]
    matrixMerch2 = matrixMerch[None, :, :]
    # compute jaccard similarity merch
    if metric == "cosine":
        # COSINE
        distMerch = 1 - ((1 + np.sum(matrixMerch1 * matrixMerch2, axis=-1) / np.linalg.norm(matrixMerch1, axis=-1) / np.linalg.norm(matrixMerch2, axis=-1)) / 2)
    elif metric == "jaccard":
        # JACCARD
        min_matrixMerch = np.minimum(matrixMerch1, matrixMerch2)
        sum_min_matrixMerch = np.sum(min_matrixMerch, axis=-1)
        
        max_matrixMerch = np.maximum(matrixMerch1, matrixMerch2)
        sum_max_matrixMerch = np.sum(max_matrixMerch, axis=-1)
        
        distMerch = 1 - (sum_min_matrixMerch / sum_max_matrixMerch)
    else:
        # L2
        distMerch = np.linalg.norm(matrixMerch1 - matrixMerch2, axis=-1)
        
        
    if fusion == "mean":
        # MEAN
        return (alpha) * routeDistance + (1-alpha) * distMerch
    elif fusion == "product":
        # PRODUCT
        return routeDistance * distMerch
    else:
        # WEIGHTED PRODUCT
        return np.power(routeDistance, alpha) * np.power(distMerch, 1-alpha)


def create_binary_matrices(routeSet1, routeSet2):
    # create binary matrix where each row represents a route
    uniqueShinglesBoth = list(set([shingle for route in routeSet1 for shingle in route[1]] + [shingle for route in routeSet2 for shingle in route[1]]))
    binaryMatrix1 = np.zeros((len(routeSet1), len(uniqueShinglesBoth)))
    binaryMatrix2 = np.zeros((len(routeSet2), len(uniqueShinglesBoth)))
    for i, route in enumerate(tqdm(routeSet1)):
        for shingle in route[1]:
            binaryMatrix1[i][uniqueShinglesBoth.index(shingle)] = 1
            
    for i, route in enumerate(tqdm(routeSet2)):
        for shingle in route[1]:
            binaryMatrix2[i][uniqueShinglesBoth.index(shingle)] = 1
    return binaryMatrix1, binaryMatrix2

def find_closest_number_with_factors(n):
    n = n+15
    factors_n = [i for i in range(1, n + 1) if n % i == 0]

    closest_number = n
    max_factors_count = len(factors_n)

    for i in range(n - 1, n-30, -1):
        factors_i = [j for j in range(1, i + 1) if i % j == 0]

        if len(factors_i) > max_factors_count:
            closest_number = i
            max_factors_count = len(factors_i)

    return closest_number

def find_num_hashes_minhash(matrix):
    if matrix.shape[1] < 125:
        num_hash_functions = find_closest_number_with_factors(matrix.shape[1])
    elif matrix.shape[1] < 500:
        num_hash_functions = find_closest_number_with_factors(matrix.shape[1]//2)
    elif matrix.shape[1] < 1000:
        num_hash_functions = find_closest_number_with_factors(matrix.shape[1]//4)
    elif matrix.shape[1] < 10_000:
        num_hash_functions = 300
    elif matrix.shape[1] < 100_000:
        num_hash_functions = 420
    else:
        num_hash_functions = 510
    return num_hash_functions

def find_threshold_lsh(matrix1, matrix2):
    tot = matrix1.shape[0]*matrix2.shape[0]
    if tot <= 100_000_000:
        threshold = 0.0
    elif tot < 500_000_000:
        threshold = 0.1
    elif tot < 1_000_000_000:
        threshold = 0.2
    elif tot < 10_000_000_000:
        threshold = 0.4
    else:
        threshold = 0.9
    return threshold


def mapping_clustroid(labels: list, cluster_idx, point_idx):
    counter = 0
    for i,label in enumerate(labels):
        if label == cluster_idx:
            if counter == point_idx:
                return i
            counter += 1

def inter_cluster_distance(sparse_distances):
    mean_distances = []
    for point in sparse_distances:
        mean_distances.append(np.mean(point))
    
    return  (1 - np.mean(mean_distances)) * sparse_distances.shape[0]

def intra_cluster_distance(sparse_distances):
    mean_distances = []
    for point in sparse_distances:
        mean_distances.append(np.mean(point))

    return mean_distances