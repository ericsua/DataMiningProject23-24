import os

import time
import math
import json
import random
import pandas as pd
import sys
import lxml
import sklearn as sk
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN, DBSCAN
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from scipy.sparse import csr_matrix, issparse, lil_matrix, coo_matrix

from tqdm import tqdm
from pandarallel import pandarallel

from numba import njit, prange, jit
from numba_progress import ProgressBar

import networkx as nx



K_SHINGLES = 3

STANDARD_FILE = 'standard_medium_new.json'
ACTUAL_FILE = 'actual_medium_new.json'

# Functions

## shingles

def hashShingles(shingles, n):
    # hash shingles
    string = "" 
    for shingle in shingles:
        string += str(shingle) + "," # [45, 4, 8] -> "45,4,8,"
    
    return hash(string) #% n

def createShingles(df, k, uniqueCities, uniqueItems, longestRoute, maxItemQuantity, permutations):
    # create shingles for each route
    shingles = []
    for index, s in df.iterrows():
        idS = s['id']
        route = s['route']
        shingle = [index]
        citiesInRoute = [] # napoli roma milano teramo bergamo [10,4,5,48,12] [10,4,5] [4,5,48] [5,48,12]
        merchandiseInRoute = np.zeros(len(uniqueItems))
        for trip in route:
            citiesInRoute.append(uniqueCities.index(trip['from']))
            #merchandiseInRoute += np.array(list(trip['merchandise'].values()))
            for item, n in trip['merchandise'].items():
                merchandiseInRoute[uniqueItems.index(item)] += n
        if len(route) > 0:
            citiesInRoute.append(uniqueCities.index(route[-1]['to']))
        if len(route) > 0:
            merchandiseInRoute = merchandiseInRoute / (maxItemQuantity*len(route))
        
        hashedShingles = []
        for i in range(len(citiesInRoute)-k+1):
            # Q: is it correct to set the modulo for the hash function to the number of permutations?
            # A: yes, because we want to have a unique hash for each shingle
            # Q: would it be better to use a different hash function?
            # A: yes, because the modulo function is not a good hash function
            hashedShingles.append(hashShingles(citiesInRoute[i:i+k], permutations) )
        
        shingle.append(np.array(hashedShingles))
        
        shingle.append(merchandiseInRoute) # quantity hot encoding
        
        shingles.append(shingle)
        
    return shingles # [ index, [shingles], [merchandise] ]

def create_shingles(s, k, uniqueCities, uniqueItems, longestRoute, maxItemQuantity, permutations):

    idS = s['id']
    route = s['route']
    shingle = [s.name]
    citiesInRoute = [] 
    merchandiseInRoute = np.zeros(len(uniqueItems))
    for trip in route:
        citiesInRoute.append(uniqueCities.index(trip['from']))
        for item, n in trip['merchandise'].items():
            merchandiseInRoute[uniqueItems.index(item)] += n
    if len(route) > 0:
        citiesInRoute.append(uniqueCities.index(route[-1]['to']))
    if len(route) > 0:
        merchandiseInRoute = merchandiseInRoute / (maxItemQuantity*len(route))
    
    hashedShingles = []
    for i in range(len(citiesInRoute)-k+1):
        hashedShingles.append(hashShingles(citiesInRoute[i:i+k], permutations))
    
    shingle.append(np.array(hashedShingles))
    shingle.append(merchandiseInRoute)
    
    return shingle

def create_shingles_selfcontained(s, k, uniqueCities, uniqueItems, longestRoute, maxItemQuantity, permutations):
    def hash_shingles(shingles):
        # hash shingles
        string = ""
        for shingle in shingles:
            string += str(shingle) + ","
        return hash(string)

    idS = s['id']
    route = s['route']
    shingle = [s.name]
    citiesInRoute = []
    merchandiseInRoute = np.zeros(len(uniqueItems))
    
    for trip in route:
        citiesInRoute.append(uniqueCities.index(trip['from']))
        for item, n in trip['merchandise'].items():
            merchandiseInRoute[uniqueItems.index(item)] += n
    
    if len(route) > 0:
        citiesInRoute.append(uniqueCities.index(route[-1]['to']))
    
    if len(route) > 0:
        merchandiseInRoute = merchandiseInRoute / (maxItemQuantity * len(route))
    
    hashedShingles = []
    
    for i in range(len(citiesInRoute) - k + 1):
        hashedShingles.append(hash_shingles(citiesInRoute[i:i + k]))
    
    shingle.append(np.array(hashedShingles))
    shingle.append(merchandiseInRoute)
    
    return shingle

## similarity computation (minhash + lsh)    

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
    if matrix.shape[1] < 1000:
        num_hash_functions = matrix.shape[1]//10
    elif matrix.shape[1] < 10_000:
        num_hash_functions = 150
    elif matrix.shape[1] < 100_000:
        num_hash_functions = 250
    else:
        num_hash_functions = 300
    return num_hash_functions

def hash_function_hash_code(num_of_hashes,n_col,next_prime):
  
    #coeffA = np.array(pick_random_coefficients(num_of_hashes,max_column_length)).reshape((num_of_hashes,1))
    #coeffB = np.array(pick_random_coefficients(num_of_hashes,max_column_length)).reshape((num_of_hashes,1))

    coeffA = np.array(random.sample(range(0,n_col*100),num_of_hashes)).reshape((num_of_hashes,1))
    coeffB = np.array(random.sample(range(0,n_col*100),num_of_hashes)).reshape((num_of_hashes,1))

    x = np.arange(n_col).reshape((1,n_col))

    hash_code = (np.matmul(coeffA,x) + coeffB) % next_prime # (num_of_hashes,n_col) so how each column index is permuted

    return hash_code

def minhash(u,num_of_hashes):
    (n_row, n_col) = u.shape
    next_prime = n_col
    hash_code = hash_function_hash_code(num_of_hashes,n_col,next_prime)

    signature_array = np.empty(shape = (n_row,num_of_hashes))

    #t2 = time.time()

    for row in tqdm(range(n_row), desc="minhashing"):
        #print("row", row)
        ones_index = np.where(u[row,:]==1)[0]
        #if len(ones_index) == 0:
        signature_array[row,:] = np.zeros((1,num_of_hashes))
            #continue
        corresponding_hashes = hash_code[:,ones_index]
        #print("ones_index", ones_index.shape, ones_index)
        #print("corresponding_hashes", corresponding_hashes.shape, corresponding_hashes)
        row_signature = np.amin(corresponding_hashes,axis=1).reshape((1,num_of_hashes))

        signature_array[row,:] = row_signature

    return signature_array

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

@jit(cache=True, nogil=True, parallel=True)
def compute_subset_similarity_matrix_only_pairs(matrix, matrixMerch, pairs, progress_proxy):
    n = matrix.shape[0]
    m = matrix.shape[1]
    similarity_pairs = np.zeros(len(pairs))
    for i in prange(len(pairs)):
        subset1 = matrix[pairs[i][0]] #replicate_row(subset_matrix, i)  
        subset2 = matrix[pairs[i][1]]
        subset1Merch = matrixMerch[pairs[i][0]]
        subset2Merch = matrixMerch[pairs[i][1]]
        
        min_matrix = np.minimum(subset1, subset2)
        sum_min_matrix = np.sum(min_matrix, axis=-1)
        
        max_matrix = np.maximum(subset1, subset2)
        sum_max_matrix = np.sum(max_matrix, axis=-1)
        
        distMerch = 1 - np.abs(np.dot(subset1Merch, subset2Merch) / (np.linalg.norm(subset1Merch) * np.linalg.norm(subset2Merch)))
        
        
        
        similarity_pairs[i] = min((1 - (sum_min_matrix / sum_max_matrix)) * distMerch, 1.0)
        if similarity_pairs[i] == 2:
            print("similarity_pairs[i]", similarity_pairs[i])
            print("dist merch", distMerch, "cosine ", np.abs(np.dot(subset1Merch, subset2Merch) / (np.linalg.norm(subset1Merch) * np.linalg.norm(subset2Merch))))
            print("dist routes", (1 - (sum_min_matrix / sum_max_matrix)))
        progress_proxy.update(1)
    return similarity_pairs

def jaccard_similarity_minhash_lsh_route_merch(matrix, matrixMerch, thresh_user=0.2):
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
    print("improved by", len(pairs) / (matrix.shape[0]*(matrix.shape[0]-1)/2)*100, "%")
    #print("num of pairs", len(pairs))
    #print("num unique i", len(set([i for i, j in pairs])))
    #print("num unique j", len(set([j for i, j in pairs])))
    #print("num unique rows", len(uniqueRows))
    #map_i = {i: index for i, index in enumerate(uniqueRowsSet)}
    #map_i_array = np.array([map_i[i] for i in range(len(map_i))])
    
    #subset_matrix = matrix[list(uniqueRowsSet)]
    
    #subset_similarity_matrix = np.full((subset_matrix.shape[0], subset_matrix.shape[0]), np.inf)
    
    print("Computing jaccard similarity on subset matrix...")
    #print("subset matrix", subset_matrix.shape)

    with ProgressBar(total=len(pairs)) as progress:
        distance_pairs = compute_subset_similarity_matrix_only_pairs(matrix, matrixMerch, pairs, progress)
        
    if len(neverSeen) > 0:
        for i, n in enumerate(neverSeen):
            distance_pairs = np.concatenate([distance_pairs, [1]*(matrix.shape[0]-1-i)])
        
        pairs = np.concatenate([pairs, np.array([[i, j] for i,n  in enumerate(neverSeen) for j in range(i, matrix.shape[0]) if i != j])])
    print("pairs", pairs.shape, pairs[-10:])
    # map back to original matrix
    print("Mapping back to original matrix...")
    # Create arrays of indices
    # Create data array for COO matrix
    indices_i, indices_j = np.array(pairs).T
    data = np.concatenate([distance_pairs, distance_pairs])

    # Create row and column index arrays for COO matrix
    rows = np.concatenate([indices_i, indices_j])
    cols = np.concatenate([indices_j, indices_i])

    # Create COO matrix
    similarity_matrix = coo_matrix((data, (rows, cols)), shape=(matrix.shape[0], matrix.shape[0]))
    
    # indices_i, indices_j = np.triu_indices(subset_similarity_matrix.shape[0], k=1)
    # similarity_matrix = similarity_matrix.tocsr()
    # # Update the similarity matrix
    # similarity_matrix[map_i_array[indices_i], map_i_array[indices_j]] = subset_similarity_matrix[indices_i, indices_j]
    # similarity_matrix[map_i_array[indices_j], map_i_array[indices_i]] = subset_similarity_matrix[indices_i, indices_j]
    
    # for i, j in tqdm(pairs, desc="lsh sim"):
    #     similarity_matrix[i, j] = np.count_nonzero(matrix[i, :] == matrix[j, :]) / matrix.shape[1]
    #     similarity_matrix[j, i] = similarity_matrix[i, j]
    
    #similarity_matrix.setdiag(1)
    similarity_matrix = similarity_matrix.tocsr()
    
    return similarity_matrix

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


def similarity_minhash_lsh_two_matrices_and_merch(matrix1, matrix1Merch, matrix2, matrix2Merch, thresh_user=0.2):
    
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
        similarity_matrix = compute_distance_pairs_merch(similarity_matrix, matrix1, matrix1Merch, matrix2, matrix2Merch, progress)
        
    return similarity_matrix

@jit(cache=True)
def compute_distance_pairs_merch(sim_matrix, matrix1, matrix1Merch, matrix2, matrix2Merch, progress_proxy):
    n = sim_matrix.shape[0]
    m = sim_matrix.shape[1]
    squareMatrix = np.full((matrix1Merch.shape[0], matrix2Merch.shape[1]), 2)
    # print("sim_matrix", sim_matrix.shape)    
    # print(numba.typeof(sim_matrix))
    # print(numba.typeof(matrix1))
    # print(numba.typeof(matrix1Merch))
    # print(numba.typeof(matrix2))
    # print(numba.typeof(matrix2Merch))
    # print(numba.typeof(progress_proxy))

    

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
        #normsSubset2Merch = np.sqrt(np.sum(np.power(subset2Merch, squareMatrix), axis=1))
        # print("subset1Merch", subset1Merch.shape)
        # print("subset2Merch", subset2Merch.shape)
        
        # COSINE
        #distMerch = 1 - (((subset1Merch * subset2Merch).sum(axis=1) / (np.sqrt(np.sum(np.power(subset1Merch, squareMatrix),axis=1)) * normsSubset2Merch)) + 1) / 2
        
        # JACCARD
        min_matrixMerch = np.minimum(subset1Merch, subset2Merch)
        sum_min_matrixMerch = np.sum(min_matrixMerch, axis=-1)

        max_matrixMerch = np.maximum(subset1Merch, subset2Merch)
        sum_max_matrixMerch = np.sum(max_matrixMerch, axis=-1)

        merch_distance = (np.divide(sum_min_matrixMerch, sum_max_matrixMerch))
        #print("merch_distance", merch_distance.shape)
        
        # L2
        # merch_distance = np.sqrt(np.sum(np.square(subset1Merch - subset2Merch), axis=-1))



        # mean
        sim_matrix[i,sim_matrix[i].nonzero()[1]] = (0.5) * route_distance + (0.5) * merch_distance
        
        # product
        #sim_matrix[i,sim_matrix[i].nonzero()[1]] = route_distance * merch_distance
        
        # weighted product
        # weightsRoutes = np.full(sim_matrix[i].nonzero()[1].shape[0], 0.8)
        # weightsMerch = np.full(sim_matrix[i].nonzero()[1].shape[0], 0.2)
        # sim_matrix[i,sim_matrix[i].nonzero()[1]] = np.power(route_distance, weightsRoutes) * np.power(merch_distance, weightsMerch)
        
        progress_proxy.update(1)
    
    return sim_matrix

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
        # print("cols", cols)
        # indexMax = np.argmax(similarities)
        # simMax = similarities[indexMax]
        # # Add the pairs to the candidate pairs
        # #candidate_pairs.extend((i, i+1+index) for index in indices)
        # candidate_pairs[i] = [indexMax, simMax]
        # similarities_actual.append(similarities)

        

    # # Create data array for COO matrix
    # data = np.concatenate([subset_sim_matrix[indices_i, indices_j], subset_sim_matrix[indices_i, indices_j]])
    
    # # Create row and column index arrays for COO matrix
    # rows = np.concatenate([indices_i_mapped, indices_j_mapped])
    # cols = np.concatenate([indices_j_mapped, indices_i_mapped])
    # print("data", data)
    # print("rows", rows)
    # print("cols", cols)

    similarity_matrix = coo_matrix((data, (rows, cols)), shape=(minhash_matrix1.shape[0], minhash_matrix2.shape[0])).tocsr()

    return similarity_matrix

def mapping_back(indices, mapping):
        new_indices = []
        for key, value in mapping.items():
            new_indices.append(indices[value])
        return new_indices
                  
def create_binary_matrix(routeSets):
    uniqueShingles = list(set(shingle for route in routeSets for shingle in route[1]))
    print("uniqueShingles", len(uniqueShingles))

    # Create a dictionary that maps each shingle to its index
    shingle_to_index = {shingle: index for index, shingle in enumerate(uniqueShingles)}
    print("shingle_to_index", len(shingle_to_index))

    binaryMatrix = np.zeros((len(routeSets), len(uniqueShingles)), dtype=int)

    for i, route in enumerate(routeSets):
        #print("i", i)
        # Get the indices of the shingles in this route
        indices = [shingle_to_index[shingle] for shingle in route[1]]
        # Use advanced indexing to set the corresponding elements in the binary matrix to 1
        binaryMatrix[i, indices] = 1

    return binaryMatrix

def mapping_clustroid(labels: list, cluster_idx, point_idx):
    counter = 0
    for i,label in enumerate(labels):
        if label == cluster_idx:
            if counter == point_idx:
                return i
            counter += 1

# Main

def main():

    def create_shingles_selfcontained(s, k, uniqueCities, uniqueItems, longestRoute, maxItemQuantity, permutations):
        import numpy as np
        def hash_shingles(shingles):
            # hash shingles
            string = ""
            for shingle in shingles:
                string += str(shingle) + ","
            return hash(string)

        idS = s['id']
        route = s['route']
        shingle = [s.name]
        citiesInRoute = []
        merchandiseInRoute = np.zeros(len(uniqueItems))
        
        for trip in route:
            citiesInRoute.append(uniqueCities.index(trip['from']))
            for item, n in trip['merchandise'].items():
                merchandiseInRoute[uniqueItems.index(item)] += n
        
        if len(route) > 0:
            citiesInRoute.append(uniqueCities.index(route[-1]['to']))
        
        if len(route) > 0:
            merchandiseInRoute = merchandiseInRoute / (maxItemQuantity * len(route))
        
        hashedShingles = []
        
        for i in range(len(citiesInRoute) - k + 1):
            hashedShingles.append(hash_shingles(citiesInRoute[i:i + k]))
        
        shingle.append(np.array(hashedShingles))
        shingle.append(merchandiseInRoute)
        return shingle
    
    HOME = os.getcwd()
    print('HOME: ',HOME)

    # # Your main program logic goes here
    # if len(sys.argv) > 1:
    #     # Check if command-line arguments are provided
    #     arg = sys.argv[1]
    #     print(f"Argument passed: {arg}")
    # else:
    #     print("No arguments provided.")

    #LOAD DATA

    # load standard and actual data
    print("\nReading standard data...")
    with open(os.path.join('data',STANDARD_FILE)) as f:
        standard = json.load(f)

    print("\nReading actual data...")
    with open(os.path.join('data', ACTUAL_FILE)) as f:
        actual = json.load(f)

    # load the data into a dataframe
    print("\nCreating standard dataframe...")
    dfStandard = pd.DataFrame(standard)
    print("\nCreating actual dataframe...")
    dfActual = pd.DataFrame(actual)

    # print head of the dataframes
    print(dfStandard.head())
    print(dfActual.head())

    # get the unique cities and items of the standard data
    cities = []
    items = []
    drivers = []
    longestRoute = 0
    shortestRoute = np.inf
    maxItemQuantity = 0

    standardRefIds = []
    for index, s in dfStandard.iterrows():
        #print(s)
        idS = s['id']
        route = s['route']
        standardRefIds.append(int(idS[1]))
        for trip in route:
            cities.append(trip['from']) 
            items.extend(trip['merchandise'].keys())
            maxItemQuantity = max(maxItemQuantity, max(trip['merchandise'].values()))
        if len(route) > 0:
            cities.append(route[-1]['to'])
            
        if len(route) > longestRoute:
            longestRoute = len(route)
            
        if len(route) < shortestRoute:
            shortestRoute = len(route)
    print("\nFinished preparing standard data")

    actualRefStandardIds = []
    for index, s in dfActual.iterrows():
        #print(s)
        idS = s['id']
        route = s['route']
        idStandard = s['sroute']
        drivers.append(s['driver'])
        actualRefStandardIds.append(int(idStandard[1]))
        for trip in route:
            cities.append(trip['from'])
            items.extend(trip['merchandise'].keys())
            maxItemQuantity = max(maxItemQuantity, max(trip['merchandise'].values()))
            
        if len(route) > 0:
            cities.append(route[-1]['to'])
            
        if len(route) > longestRoute:
            longestRoute = len(route)
        
        if len(route) < shortestRoute:
            shortestRoute = len(route)
    print("\nFinished preparing actual data")

    # find the unique cities and items
    uniqueCities = sorted(list(set(cities)))
    #uniqueCities.insert(0, 'NULL')          # add NULL city, for padding vectors with different lengths (trips in routes)
    uniqueItems = sorted(list(set(items)))
    # find the unique drivers
    uniqueDrivers = sorted(list(set(drivers)))

    print("\nSorted cities and items")

    if shortestRoute < 2:
        K_SHINGLES = 2

    threeShingles = []

    for i, c1 in enumerate(uniqueCities):
        for j, c2 in enumerate(uniqueCities):
            if i == j:
                continue
            for k, c3 in enumerate(uniqueCities):
                if j == k or i == k:
                    continue
                threeShingles.append([c1, c2, c3])
                
    permutations = math.perm(len(uniqueCities), K_SHINGLES)

    print("\nComputed all possible three-shingles")

    print("\nUnique cities: ", uniqueCities)
    print("Unique items: ", uniqueItems)
    print("Unique drivers: ", uniqueDrivers)

    print("\nNumber of cities: ", len(uniqueCities))
    print("Number of items: ", len(uniqueItems))
    print("Number of drivers: ", len(uniqueDrivers))

    print("\nLongest route: ", longestRoute)
    print("Shortest route: ", shortestRoute)

    print("\nMax item quantity: ", maxItemQuantity)

    print("\nNumber of three-shingles: ", len(threeShingles))

    print(f"\n{K_SHINGLES}-shingles: ", math.perm(len(uniqueCities), K_SHINGLES))
    print(f"{K_SHINGLES}-shingles: ", math.comb(len(uniqueCities), K_SHINGLES))

    print(f"\n\033[92mK-Shingles used: {K_SHINGLES} \033[0m")

    # SHINGLES CREATION

    #standardSets = createShingles(dfStandard, k=K_SHINGLES, uniqueCities=uniqueCities, uniqueItems=uniqueItems, longestRoute=longestRoute, maxItemQuantity=maxItemQuantity, permutations=permutations)
    #actualSets = createShingles(dfActual, k=K_SHINGLES, uniqueCities=uniqueCities, uniqueItems=uniqueItems, longestRoute=longestRoute, maxItemQuantity=maxItemQuantity, permutations=permutations)
    pandarallel.initialize(progress_bar=True)
    standardSets = dfStandard.parallel_apply(lambda s: create_shingles_selfcontained(s, K_SHINGLES, uniqueCities, uniqueItems, longestRoute, maxItemQuantity, permutations), axis=1)
    standardSets = standardSets.tolist()
    actualSets = dfActual.parallel_apply(lambda s: create_shingles_selfcontained(s, K_SHINGLES, uniqueCities, uniqueItems, longestRoute, maxItemQuantity, permutations), axis=1)
    actualSets = actualSets.tolist()

    print("\nstandardSets", len(standardSets), "shape first element", standardSets[0][1].shape, standardSets[0])
    print("\nactualSets", len(actualSets),  "shape first element", standardSets[0][1].shape, actualSets[0])

    print("\nstandardSets:", len(standardSets))
    print("actualSets:", len(actualSets))

    assert len(standardSets[0]) == 3, "The length of the standard set is not equal to 3 (index, shingles, merchandise)"
    assert len(standardSets[0][2]) == len(uniqueItems), "The length of the merchandise vector is not equal to the number of unique items"


    # TASK 1
    # SIMILARITY COMPUTATION

    # convert routes and merchandise to binary matrices
    # binary matrix where each row represents a route
    print("Creating route binary matrix...")
    route_matrix, route_matrix_standard = create_binary_matrices(actualSets, standardSets)
    print("\nroute_matrix actual", route_matrix.shape, route_matrix[0])
    print("\nroute_matrix standard", route_matrix_standard.shape, route_matrix_standard[0])

    print("Minhashing route matrix...")    
    num_hash_functions = find_num_hashes_minhash(route_matrix)
    #route_matrix = minhash(route_matrix, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)
    route_matrix, route_matrix_standard = minhash_matrices(route_matrix, route_matrix_standard, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)
    print("\nroute_matrix minhash", route_matrix.shape, route_matrix[0])
    # binary matrix where each row represents merchandise

    print("Creating merchandise binary matrix...")
    merch_matrix = np.array([s[2] for s in actualSets])

    print("\nmerch_matrix", merch_matrix.shape, merch_matrix)
    print("merch_matrix contains nan", np.isnan(merch_matrix).any())

    # compute Jaccard similarity for each matrix
    # print("Computing Jaccard similarity route matrix...")
    # route_similarity = jaccard_similarity_minhash_lsh(route_matrix, thresh_user=0.4)
    # #route_similarity = jaccard_similarity_matrix(route_matrix)
    # print("\nroute_similarity", type(route_similarity), route_similarity.shape,route_similarity[0, 0], route_similarity[0])
    # #merch_similarity = jaccard_similarity_matrix_merch(merch_matrix)
    # print("Computing Jaccard similarity merchandise matrix...")
    # #merch_similarity = similarity_matrix_merch(merch_matrix)
    # merch_similarity_lsh = jaccard_similarity_minhash_lsh(merch_matrix, thresh_user=0.4)
    # print("\nmerch_similarity", type(merch_similarity_lsh), merch_similarity_lsh.shape, merch_similarity_lsh[0])

    print("Computing Jaccard similarity route matrix...")
    actualSetsDistances, map_indices_back = jaccard_similarity_minhash_lsh_route_merch(route_matrix, merch_matrix, thresh_user=0.0)
    #route_similarity = jaccard_similarity_matrix(route_matrix)
    print("\nactualSetsDistances", type(actualSetsDistances), actualSetsDistances.shape,actualSetsDistances[0, 0], actualSetsDistances[0])
    print("map indices back", map_indices_back)


    # # compute final Jaccard distance
    # print("Multiplying Jaccard similarities...")
    # actualSetsDistances = (route_similarity.multiply(merch_similarity_lsh))
    # actualSetsDistances = np.nan_to_num(actualSetsDistances, nan=0)
    #actualSetsDistances = 1 - actualSetsDistances
    #print("\nactualSetsDistances", actualSetsDistances.shape, actualSetsDistances[0, 0], actualSetsDistances[0])

    # TASK 2
    print("\n\nTASK 2 ESSENTIALS\n\n")

    # standardToActualSetsDistances = None
    #route_matrix_standard = create_binary_matrix(standardSets)
    print("Minhashing standard route matrix...")
    print("\nroute_matrix_standard", route_matrix_standard.shape, route_matrix_standard[0])
    #route_matrix_standard = minhash(route_matrix_standard, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)
    print("\nroute_matrix_standard minhash", route_matrix_standard.shape, route_matrix_standard[0])

    merch_matrix_standard = np.array([s[2] for s in standardSets])

    route_similarity_standard_to_actual = similarity_minhash_lsh_two_matrices_and_merch(route_matrix, merch_matrix, route_matrix_standard, merch_matrix_standard, thresh_user=0.0)
    print("\nroute_similarity_standard_to_actual", route_similarity_standard_to_actual.shape, route_similarity_standard_to_actual[0])

    ## TASK 3
    ## Dictionary containing all the Sets [idx, [shingles], [quantity-hot merch]] for all the drivers
    actualDriverSets = {}
    for driver in uniqueDrivers:
        dfActualDriver = dfActual[dfActual['driver'] == driver]
        actualDriverSets[driver] = createShingles(dfActualDriver, k=K_SHINGLES, uniqueCities=uniqueCities, uniqueItems=uniqueItems, longestRoute=longestRoute, maxItemQuantity=maxItemQuantity, permutations=permutations)

    ## Dictionary containing the distance matrix for all the drivers
    driversDistances = {}
    driversIndicesBack = {}
    for driver, driverSet in actualDriverSets.items():
        
        driver_indices = [i[0] for i in driverSet]
        route_matrix = create_binary_matrix(driverSet)

        num_hash_functions = find_num_hashes_minhash(route_matrix)
        route_matrix = minhash(route_matrix, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)

        merch_matrix = np.array([s[2] for s in driverSet])

        actualSetsDistances, mapping= jaccard_similarity_minhash_lsh_route_merch(route_matrix, merch_matrix, thresh_user=0.2)

        driver_indices = mapping_back(driver_indices, mapping)

        driversDistances[driver] = actualSetsDistances
        driversIndicesBack[driver] = driver_indices


    forward_expansion = len(dfActual) // len(dfStandard)
    best_clustroids = []

    driversClustersPoints = {}
    driversLabels = {}
    driversClusterLabels = {}
    driversIdealPoint = {}
    driversIdealIndex = {}

    for driver, driverDistance in driversDistances.items():
        print('performing DBSCAN for driver ', driver)

        #print(f'best min {best_min} best max {best_max}')
        hdb = HDBSCAN(min_cluster_size=2, max_cluster_size=forward_expansion, metric="precomputed", store_centers=None,allow_single_cluster=False ).fit(driverDistance.copy())
        print(f"num clusters found for driver {driver}: {len(set(hdb.labels_))}")
        print('labels: ', hdb.labels_)
        labels = hdb.labels_

        ## Selecting the cluster that has the higher intra-cluster similarity
        # Compute pairwise similarity/distance matrix within clusters
        pairwise_similarities = []
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = driverDistance[labels == label]
            pairwise_similarity = 1 - pairwise_distances(cluster_points, metric='euclidean')
            
            pairwise_similarities.append(pairwise_similarity)
        # Compute the average similarity within each cluster
        avg_similarities = [np.mean(similarity) for similarity in pairwise_similarities]

        # Select the cluster with the highest average similarity
        selected_cluster = np.argmax(avg_similarities)
        #print('selected cluster',selected_cluster)
        # Access the data points in the selected cluster
        selected_cluster_points = driverDistance[labels == selected_cluster]

        # Find the medoid that maximize the inter-cluster similarity
        pairwise_similarity = 1 - pairwise_distances(selected_cluster_points, metric='euclidean')
        selected_point = np.argmax(np.sum(pairwise_similarity, axis=0))
        #print('point inside the clustroid: ', selected_point)
        best_clustroid = mapping_clustroid(labels, selected_cluster, selected_point)
        best_clustroids.append(best_clustroid)
        #print('label index of the best point: ',best_clustroid)
        best_index = driversIndicesBack[driver][best_clustroid]
        #print('index in the dataframe of the actual route: ', best_index)
        selected_point = selected_cluster_points[selected_point]
        #print(selected_point)

        driversClustersPoints[driver] = selected_cluster_points
        driversLabels[driver] = labels
        driversClusterLabels[driver] = selected_cluster
        driversIdealPoint[driver] = selected_point
        driversIdealIndex[driver] = best_index

if __name__ == "__main__":
    main()
