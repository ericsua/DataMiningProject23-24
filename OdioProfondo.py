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

    # Your main program logic goes here
    if len(sys.argv) > 1:
        # Check if command-line arguments are provided
        arg = sys.argv[1]
        print(f"Argument passed: {arg}")
    else:
        print("No arguments provided.")

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

    print("\nNumber of cities: ", len(uniqueCities))
    print("Number of items: ", len(uniqueItems))

    print("\nLongest route: ", longestRoute)
    print("Shortest route: ", shortestRoute)

    print("\nMax item quantity: ", maxItemQuantity)

    print("\nNumber of three-shingles: ", len(threeShingles))

    print(f"\n{K_SHINGLES}-shingles: ", math.perm(len(uniqueCities), K_SHINGLES))
    print(f"{K_SHINGLES}-shingles: ", math.comb(len(uniqueCities), K_SHINGLES))

    print(f"\n\033[92mK-Shingles used: {K_SHINGLES} \033[0m")

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

    # convert routes and merchandise to binary matrices
    # binary matrix where each row represents a route
    print("Creating route binary matrix...")
    route_matrix, route_matrix_standard = create_binary_matrices(actualSets, standardSets)
    print("\nroute_matrix actual", route_matrix.shape, route_matrix[0])
    print("\nroute_matrix standard", route_matrix_standard.shape, route_matrix_standard[0])

    print("Minhashing route matrix...")    
    num_hash_functions = find_num_hashes_minhash(route_matrix)
    route_matrix = minhash(route_matrix, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)
    print("\nroute_matrix minhash", route_matrix.shape, route_matrix[0])
    # binary matrix where each row represents merchandise

    print("Creating merchandise binary matrix...")
    merch_matrix = np.array([s[2] for s in actualSets])

    print("\nmerch_matrix", merch_matrix.shape, merch_matrix)
    print("merch_matrix contains nan", np.isnan(merch_matrix).any())

    # compute Jaccard similarity for each matrix
    print("Computing Jaccard similarity route matrix...")
    route_similarity = jaccard_similarity_minhash_lsh(route_matrix, thresh_user=0.4)
    #route_similarity = jaccard_similarity_matrix(route_matrix)
    print("\nroute_similarity", route_similarity.shape, route_similarity[0])
    #merch_similarity = jaccard_similarity_matrix_merch(merch_matrix)
    print("Computing Jaccard similarity merchandise matrix...")
    merch_similarity = similarity_matrix_merch(merch_matrix)
    print("\nmerch_similarity", merch_similarity.shape, merch_similarity[0])

    # compute final Jaccard distance
    actualSetsDistances = route_similarity * merch_similarity
    actualSetsDistances = np.nan_to_num(actualSetsDistances, nan=0)
    actualSetsDistances = 1 - actualSetsDistances
    print("\nactualSetsDistances", actualSetsDistances.shape, actualSetsDistances[0])

    standardToActualSetsDistances = None
    #route_matrix_standard = create_binary_matrix(standardSets)
    print("\nroute_matrix_standard", route_matrix_standard.shape, route_matrix_standard[0])
    route_matrix_standard = minhash(route_matrix_standard, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)
    print("\nroute_matrix_standard minhash", route_matrix_standard.shape, route_matrix_standard[0])

    merch_matrix_standard = np.array([s[2] for s in standardSets])

    route_similarity_standard_to_actual = jaccard_similarity_minhash_lsh_two_matrices(route_matrix, route_matrix_standard, thresh_user=0.2)
    print("\nroute_similarity_standard_to_actual", route_similarity_standard_to_actual.shape, route_similarity_standard_to_actual[0])






if __name__ == "__main__":
    main()
