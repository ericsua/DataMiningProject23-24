from numba import njit, prange, jit
from numba_progress import ProgressBar

import numpy as np

import random
from tqdm import tqdm

from scipy.sparse import csr_matrix, coo_matrix


def hash_function_hash_code(num_of_hashes,n_col,next_prime):
  
    #coeffA = np.array(pick_random_coefficients(num_of_hashes,max_column_length)).reshape((num_of_hashes,1))
    #coeffB = np.array(pick_random_coefficients(num_of_hashes,max_column_length)).reshape((num_of_hashes,1))

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

    #t2 = time.time()

    for row in tqdm(range(n_row), desc="minhashing"):
        #print("row", row)
        ones_index = np.where(matrix1[row,:]==1)[0]
        #if len(ones_index) == 0:
        signature1_array[row,:] = np.zeros((1,num_of_hashes))

            #continue
        corresponding_hashes = hash_code[:,ones_index]
        #print("ones_index", ones_index.shape, ones_index)
        #print("corresponding_hashes", corresponding_hashes.shape, corresponding_hashes)
        row_signature = np.amin(corresponding_hashes,axis=1).reshape((1,num_of_hashes))

        signature1_array[row,:] = row_signature

    for row in tqdm(range(matrix2.shape[0]), desc="minhashing second matrix"):
        #print("row", row)
        ones_index = np.where(matrix2[row,:]==1)[0]
        #if len(ones_index) == 0:
        signature2_array[row,:] = np.zeros((1,num_of_hashes))

            #continue
        corresponding_hashes = hash_code[:,ones_index]
        #print("ones_index", ones_index.shape, ones_index)
        #print("corresponding_hashes", corresponding_hashes.shape, corresponding_hashes)
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
    #bands = b
        
    print("final bands", b)
    signature_matrix = np.full((minhash_matrix.shape[0], b), np.inf)
    
    # if threshold is 0.8,
    threshold = (1 / b) ** (1 / r) 
    print("lsh threshold", threshold)
    
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
    
    # Generate the hash functions
    # hash_functions = [lambda x, a=a, b=b: (a * x + b) % minhash_matrix.shape[1] for a, b in zip(random.sample(range(1000), bands), random.sample(range(1000), bands))]
    # hash_function = lambda x: hash(",".join([str(x[i]) for i in range(len(x))]))
    
    def hash_function(x):
        # print("x",x)
        var = hash(",".join([str(x[i]) for i in range(len(x))]))
        # print ("str x ", (",".join([(x[i]) for i in range(len(x))])))
        # print ("var", var)
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
        
    print("final bands", b)
    signature_matrix1 = np.full((minhash_matrix1.shape[0], b), np.inf)
    signature_matrix2 = np.full((minhash_matrix2.shape[0], b), np.inf)
    

    threshold = (1 / b) ** (1 / r) 
    print("lsh threshold", threshold)
    
    # # For each band
    # print("Computing hash values of bands...")
    # hash_values1 = np.apply_along_axis(lambda x: hash_function(x) % minhash_matrix1.shape[0], 1, minhash_matrix1.reshape(-1, r))
    # print("hash_values1", hash_values1.shape, hash_values1)
    # hash_values2 = np.apply_along_axis(lambda x: hash_function(x) % minhash_matrix2.shape[0], 1, minhash_matrix2.reshape(-1, r))
    # print("hash_values2", hash_values2.shape, hash_values2)

    print("minhash_matrix1.reshape(-1, r).shape",minhash_matrix1.reshape(-1, r).shape)

    # For each band
    print("Computing hash values of bands...")
    hash_values1 = np.apply_along_axis(hash_function, 1, minhash_matrix1.reshape(-1, r))
    # print("hash_values1", hash_values1.shape, hash_values1)
    hash_values2 = np.apply_along_axis(hash_function, 1, minhash_matrix2.reshape(-1, r))
    # print("hash_values2", hash_values2.shape, hash_values2)


    # Reshape the hash values to match the signature matrix
    hash_values1 = hash_values1.reshape(minhash_matrix1.shape[0], b)
    # print("hash_values1", hash_values1.shape, hash_values1)
    hash_values2 = hash_values2.reshape(minhash_matrix2.shape[0], b)
    # print("hash_values2", hash_values2.shape, hash_values2) 
    # Update the signature matrix
    signature_matrix1 = hash_values1
    signature_matrix2 = hash_values2
    
    
    # find candidate pairs
    print("Finding candidate pairs...")
    # similarities_actual=[]
    # candidate_pairs = np.empty((minhash_matrix1.shape[0], 2))

    data=[]
    rows=[]
    cols=[]

    for i in tqdm(range(signature_matrix1.shape[0])):
        # Compute the similarity of the current row with all following rows
        similarities = np.sum(signature_matrix2 == signature_matrix1[i, :], axis=1) / b
        # print("similarities", similarities.shape, similarities)
        # Find the indices of the rows that have a similarity greater than or equal to the threshold
        indices = np.nonzero(similarities >= threshold)[0]
        # print("indices", indices.shape, indices)

        # print("similarities[indices] ",similarities[indices])

        data.extend(similarities[indices])
        # print("data", data)
        rows.extend([i]*len(indices))
        # print("rows", rows)
        cols.extend(indices)


    similarity_matrix = coo_matrix((data, (rows, cols)), shape=(minhash_matrix1.shape[0], minhash_matrix2.shape[0])).tocsr()

    return similarity_matrix

@jit(cache=True)
def compute_distance_pairs_merch(sim_matrix, matrix1, matrix1Merch, matrix2, matrix2Merch, progress_proxy, metric="jaccard", alpha=0.8, fusion="mean"):
    n = sim_matrix.shape[0]
    m = sim_matrix.shape[1]
    squareMatrix = np.full((matrix1Merch.shape[0], matrix2Merch.shape[1]), 2)


    for i in prange(n):
        subset1 = matrix1[i].reshape(1, -1) #replicate_row(subset_matrix, i) 
        # print("subset1", subset1.shape)
        subset2 = matrix2[sim_matrix[i].nonzero()[1]]
        # print("subset2", subset2.shape)
        min_matrix = np.minimum(subset1, subset2)
        sum_min_matrix = np.sum(min_matrix, axis=-1)
        
        max_matrix = np.maximum(subset1, subset2)
        sum_max_matrix = np.sum(max_matrix, axis=-1)

        route_distance = (np.divide(sum_min_matrix, sum_max_matrix))
        #print("route_distance", route_distance.shape)

        subset1Merch = matrix1Merch[i].reshape(1, -1) #replicate_row(subset_matrixMerch, i)
        subset2Merch = matrix2Merch[sim_matrix[i].nonzero()[1]]
        
        # print("subset1Merch", subset1Merch.shape)
        # print("subset2Merch", subset2Merch.shape)
        
        if metric == "cosine":
            # COSINE
            normsSubset2Merch = np.sqrt(np.sum(np.power(subset2Merch, squareMatrix), axis=1))
            merch_distance = 1 - (((subset1Merch * subset2Merch).sum(axis=1) / (np.sqrt(np.sum(np.power(subset1Merch, squareMatrix),axis=1)) * normsSubset2Merch)) + 1) / 2
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
            merch_distance = np.sqrt(np.sum(np.square(subset1Merch - subset2Merch), axis=-1))


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
    # print("similarity_matrix", similarity_matrix.shape, similarity_matrix)

    # uniqueRowsSet = set([i for i, j in pairs] + [j for i, j in pairs]) # (1,2) (1,4) (1,5)
    # neverSeen = set([i for i in range(matrix1.shape[0])]) - uniqueRowsSet
    

    # sortedUniqueRowsSet = sorted(list(uniqueRowsSet))
    # print("sortedUniqueRowsSet", sortedUniqueRowsSet)

    # subset_matrix1 = matrix1[sortedUniqueRowsSet]
    # subset_matrix1Merch = matrix1Merch[sortedUniqueRowsSet]
    # print("subset_matrix1", subset_matrix1.shape, subset_matrix1[0])
    # print("subset_matrix1Merch", subset_matrix1Merch.shape, subset_matrix1Merch[0])

    # subset_matrix2 = matrix2[sortedUniqueRowsSet]
    # subset_matrix2Merch = matrix2Merch[sortedUniqueRowsSet]
    # print("subset_matrix2", subset_matrix2.shape, subset_matrix2[0])
    # print("subset_matrix2Merch", subset_matrix2Merch.shape, subset_matrix2Merch[0])

    # subset_similarity_matrix = np.full((subset_matrix1.shape[0], subset_matrix2.shape[0]), np.inf)
        
    print("Computing distance  on subset matrix...")
    with ProgressBar(total=matrix1.shape[0]) as progress:
        similarity_matrix = compute_distance_pairs_merch(similarity_matrix, matrix1, matrix1Merch, matrix2, matrix2Merch, progress, metric="jaccard", alpha=0.8, fusion="mean")
        
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
    print("n", n, "m", m)
    print("matrix merch", matrixMerch.shape)
    print("matrix square", squareMatrix.shape)
    print("routeWeights", routeWeights.shape)
    print("merchWeights", merchWeights.shape)
    normsSubset2Merch = np.sqrt(np.sum(np.power(subset2Merch, squareMatrix), axis=1))
    for i in prange(n):
        subset1 = matrix[i].reshape(1, -1) #replicate_row(subset_matrix, i)  
        subset1Merch = matrixMerch[i].reshape(1, -1)
        #print("subset1", subset1.shape)
        #print("subset2", subset2.shape)
        
        min_matrix = np.minimum(subset1, subset2)
        sum_min_matrix = np.sum(min_matrix, axis=-1)
        
        max_matrix = np.maximum(subset1, subset2)
        sum_max_matrix = np.sum(max_matrix, axis=-1)
        
        #print("sum_min_matrix", sum_min_matrix.shape)
        
        #print("merch1", subset1Merch.shape)
        #print("merch2", subset2Merch.shape)
        
        #distMerch = 1 - np.abs(np.dot(subset1Merch, subset2Merch.T) / (np.linalg.norm(subset1Merch) * np.linalg.norm(subset2Merch)))
        
        
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
        # if i == 0 or i == n-1:
        #     print("i", i, "distMerch", distMerch.shape, distMerch)
        #     print("i", i, "sum_min_matrix", (1 - (sum_min_matrix / sum_max_matrix)).shape, (1 - (sum_min_matrix / sum_max_matrix)))
        #     print("i", i, "prod", ((1 - (sum_min_matrix / sum_max_matrix)) * distMerch).shape, ((1 - (sum_min_matrix / sum_max_matrix)) * distMerch))
        #print(i, (1 - (sum_min_matrix / sum_max_matrix)) * distMerch)
        
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
    #similarity_matrix = csr_matrix((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
    #similarity_matrix = lil_matrix((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
    pairs = lsh(matrix, thresh_user=thresh_user)
    #uniqueRows = np.unique([i for i, j in pairs] + [j for i, j in pairs])
    uniqueRowsSet = set([i for i, j in pairs] + [j for i, j in pairs]) # (1,2) (1,4) (1,5)
    neverSeen = set([i for i in range(matrix.shape[0])]) - uniqueRowsSet
    print("neverSeen", neverSeen)
    #print("uniqueRows numpy", len(uniqueRows))
    print("num of subset of rows to check similarity:", len(uniqueRowsSet))
    #print(" num of pairs", len(uniqueRowsSet)*(len(uniqueRowsSet)-1)/2)
    print(" num of pairs", len(pairs))
    print(" instead of", matrix.shape[0]*(matrix.shape[0]-1)/2)
    print("improved by", (1 - len(pairs) / (matrix.shape[0]*(matrix.shape[0]-1)/2)) *100, "%")
    
        
    print("Computing jaccard similarity on subset matrix...")
    #print("subset matrix", subset_matrix.shape)

    # with ProgressBar(total=len(pairs)) as progress:
    #     distance_pairs = compute_subset_similarity_matrix_only_pairs(matrix, matrixMerch, pairs, progress)
    
    sortedUniqueRowsSet = sorted(list(uniqueRowsSet))
    subset_matrix = matrix[sortedUniqueRowsSet]
    subset_matrixMerch = matrixMerch[sortedUniqueRowsSet]
    print("subset_matrix", subset_matrix.shape, subset_matrix[0])
    print("subset_matrixMerch", subset_matrixMerch.shape, subset_matrixMerch[0])
    with ProgressBar(total=len(sortedUniqueRowsSet)) as progress:
        subset_sim_matrix = compute_subset_similarity_matrix_and_merch(subset_matrix, subset_matrixMerch, progress, metric=metric, alpha=alpha, fusion=fusion)
    print("subset_sim_matrix", subset_sim_matrix.shape, subset_sim_matrix[0])
    print("subset_sim_matrix contains nan", np.isnan(subset_sim_matrix).any())
    print("nan indices", len(np.argwhere(np.isnan(subset_sim_matrix))), np.argwhere(np.isnan(subset_sim_matrix)))
    
    # if len(neverSeen) > 0:
    #     for i, n in enumerate(neverSeen):
    #         distance_pairs = np.concatenate([distance_pairs, [1]*(matrix.shape[0]-1-i)])
        
    #     pairs = np.concatenate([pairs, np.array([[i, j] for i,n  in enumerate(neverSeen) for j in range(i, matrix.shape[0]) if i != j])])
    #print("pairs", pairs.shape, pairs[-10:])
    # map back to original matrix
    print("Mapping back to original matrix...")
    
    lenMatrixNoNeverSeen = matrix.shape[0] - len(neverSeen)
    
    # remove never seen rows and map indices
    map_indices = {}
    sortedNeverSeen = sorted(list(neverSeen))
    counter = 0
    for i in range(matrix.shape[0]):
        if i in sortedNeverSeen:
            continue
        map_indices[i] = counter
        counter += 1
        
    print("map_indices", map_indices)
    map_indices_back = {v: k for k, v in map_indices.items()}
    

    #similarity_matrix.setdiag(1)
    subset_sim_matrix = csr_matrix(subset_sim_matrix)
    
    return subset_sim_matrix, map_indices, map_indices_back


def create_binary_matrices(routeSet1, routeSet2):
    # create binary matrix where each row represents a route
    uniqueShinglesBoth = list(set([shingle for route in routeSet1 for shingle in route[1]] + [shingle for route in routeSet2 for shingle in route[1]]))
    binaryMatrix1 = np.zeros((len(routeSet1), len(uniqueShinglesBoth)))
    binaryMatrix2 = np.zeros((len(routeSet2), len(uniqueShinglesBoth)))
    for i, route in enumerate(routeSet1):
        for shingle in route[1]:
            binaryMatrix1[i][uniqueShinglesBoth.index(shingle)] = 1
            
    for i, route in enumerate(routeSet2):
        for shingle in route[1]:
            binaryMatrix2[i][uniqueShinglesBoth.index(shingle)] = 1
    return binaryMatrix1, binaryMatrix2

def find_num_hashes_minhash(matrix):
    if matrix.shape[1] < 150:
        num_hash_functions = matrix.shape[1]
    elif matrix.shape[1] < 500:
        num_hash_functions = matrix.shape[1]//2
    elif matrix.shape[1] < 1000:
        num_hash_functions = matrix.shape[1]//10
    elif matrix.shape[1] < 10_000:
        num_hash_functions = 150
    elif matrix.shape[1] < 100_000:
        num_hash_functions = 250
    else:
        num_hash_functions = 300
    return num_hash_functions

def find_threshold_lsh(matrix1, matrix2):
    tot = matrix1.shape[0]*matrix2.shape[0]
    if tot < 100_000_000:
        threshold = 0.0
    elif tot < 1_000_000_000:
        threshold = 0.2
    elif tot < 10_000_000_000:
        threshold = 0.4
    else:
        threshold = 0.7
    return threshold