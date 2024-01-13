import os
import sys
HOME = os.path.dirname(os.path.realpath(__file__))
print('HOME: ',HOME)
## add utils to path

sys.path.append(HOME)
# sys.path.append(os.path.join(HOME, 'utils'))
# sys.path.append(os.path.join(HOME, 'data'))
# sys.path.append(os.path.join(HOME, 'results'))

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


from utils.shingles import *
from utils.clustering import *


import umap


### GLOBAL VARIABLES ###


# STANDARD_FILE = 'standard_small_order_printClustroids.json'
# ACTUAL_FILE = 'actual_small_order_printClustroids.json'

# STANDARD_FILE = 'standard_small.json'
# ACTUAL_FILE = 'actual_small.json'

STANDARD_FILE = 'standard_medium_last.json'
ACTUAL_FILE = 'actual_medium_last.json'

# STANDARD_FILE = 'standard_tiny.json'
# ACTUAL_FILE = 'actual_tiny.json'

# STANDARD_FILE = 'standard_big_new_3.json'
# ACTUAL_FILE = 'actual_big_new_3.json'

K_SHINGLES = 3

ALPHA = 0.5         # weight of route vector wrt merchandise vector, between 0 and 1
METRIC = "jaccard"  # "cosine" or "jaccard" or "L2", metric used for merchandise vectors
FUSION = "mean"     # "mean" or "product" or "weigthed product", how to fuse the route and merchandise vectors 
                    # (e.g. mean: (ALPHA) * route + (1-ALPHA) * merchandise, product: route * merchandise, weighted product: route * (merchandise + 1))




#### READ DATA ####

# load standard and actual data
print("\nReading standard data...")
with open(os.path.join(HOME,'data',STANDARD_FILE)) as f:
    standard = json.load(f)

print("\nReading actual data...")
with open(os.path.join(HOME, 'data', ACTUAL_FILE)) as f:
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
    actualRefStandardIds.append(int(idStandard[1]))
    drivers.append(s['driver'])
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

standardIds = dfStandard['id'].tolist()
print("standardIds: ", standardIds)

print("\nNumber of cities: ", len(uniqueCities))
print("Number of items: ", len(uniqueItems))

print("\nLongest route: ", longestRoute)
print("Shortest route: ", shortestRoute)

print("\nMax item quantity: ", maxItemQuantity)

print("\nNumber of three-shingles: ", len(threeShingles))

print(f"\n{K_SHINGLES}-shingles: ", math.perm(len(uniqueCities), K_SHINGLES))
print(f"{K_SHINGLES}-shingles: ", math.comb(len(uniqueCities), K_SHINGLES))

print(f"\n\033[92mK-Shingles used: {K_SHINGLES} \033[0m")



#############################################




##### CREATE SHINGLES #####

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








######## ESSENTIALS ########

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


print("Computing Jaccard similarity route matrix...")
threshold_lsh = find_threshold_lsh(route_matrix, route_matrix)
actualSetsDistances, map_indices, map_indices_back = jaccard_similarity_minhash_lsh_route_merch(route_matrix, merch_matrix, thresh_user=threshold_lsh, metric=METRIC, fusion=FUSION, alpha=ALPHA)
#route_similarity = jaccard_similarity_matrix(route_matrix)
print("\nactualSetsDistances", type(actualSetsDistances), actualSetsDistances.shape,actualSetsDistances[0, 0], actualSetsDistances[0])
print("map indices back", map_indices_back)



####### ESSENTIALS FOR TASK 2 ########
print("\n\nTASK 2 ESSENTIALS\n\n")

# standardToActualSetsDistances = None
#route_matrix_standard = create_binary_matrix(standardSets)
print("Minhashing standard route matrix...")
print("\nroute_matrix_standard", route_matrix_standard.shape, route_matrix_standard[0])
#route_matrix_standard = minhash(route_matrix_standard, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)
print("\nroute_matrix_standard minhash", route_matrix_standard.shape, route_matrix_standard[0])

merch_matrix_standard = np.array([s[2] for s in standardSets])

threshold_lsh_actual_to_standard = find_threshold_lsh(route_matrix, route_matrix_standard)
route_similarity_standard_to_actual = similarity_minhash_lsh_two_matrices_and_merch(route_matrix, merch_matrix, route_matrix_standard, merch_matrix_standard, thresh_user=threshold_lsh_actual_to_standard, metric=METRIC, fusion=FUSION, alpha=ALPHA)
print("\nroute_similarity_standard_to_actual", route_similarity_standard_to_actual.shape, route_similarity_standard_to_actual[0])

#merch_similarity_lsh_standard_to_actual = jaccard_similarity_minhash_lsh_two_matrices(merch_matrix, merch_matrix_standard, thresh_user=0.4)



###### CLUSTERING ######

# HDBSCAN clustering

# compute mean forward expansion
forward_expansion = len(actualSets) // len(standardSets)
print("forward_expansion", forward_expansion)

print("actualSetsDistances", actualSetsDistances.shape, actualSetsDistances)


print("type(actualSetsDistances)", type(actualSetsDistances), actualSetsDistances.dtype, actualSetsDistances.shape, actualSetsDistances.count_nonzero(), min(actualSetsDistances.getnnz(axis=-1)), np.unique(actualSetsDistances.data))
#actualSetsDistances = np.array(actualSetsDistances)

#print("get nnz of 2 values", np.where((actualSetsDistances==2))[0])


print("Computing HDBSCAN...")
hdb = HDBSCAN(min_cluster_size=forward_expansion//3, max_cluster_size=forward_expansion, metric="precomputed", store_centers=None,allow_single_cluster=False).fit(actualSetsDistances.copy())
#hdb = DBSCAN(eps=0.5, min_samples=forward_expansion//3, metric="precomputed").fit(actualSetsDistances.copy())

labels_HDBSCAN = hdb.labels_
print("num clusters found: ", len(set(labels_HDBSCAN)))
print("biggest cluster: ", max(labels_HDBSCAN, key=list(labels_HDBSCAN).count), " num elements: ", list(labels_HDBSCAN).count(max(labels_HDBSCAN, key=list(labels_HDBSCAN).count)))

# Create a color map that maps each unique label to a color
unique_labels = np.unique(labels_HDBSCAN)
#unique_labels = unique_labels[unique_labels != -1]
print("unique_labels: ", len(unique_labels), unique_labels, "standard sets len", len(standardSets))

#centroids = hdb.centroids_
#medoids = hdb.medoids_

#print("centroids: ", centroids.shape)
#print("medoids: ", medoids.shape)

# find the medoids using the clusters found by HDBSCAN
print("actualSetsDistances: ", actualSetsDistances.shape, actualSetsDistances[0])
medoidsIndices = []
cluster_mean_distances = []
for cluster in unique_labels:
    #print("cluster: ", cluster)
    if cluster in [-1, -2, -3]:
        continue
    cluster_elements = np.where(labels_HDBSCAN == cluster)[0]
    #print("cluster_elements: ", cluster_elements.shape, cluster_elements)
    #print("cluster_elements: ", actualSetsDistances[cluster_elements].shape)
    cluster_distances = actualSetsDistances[cluster_elements][:,cluster_elements]
    #print("cluster_distances: ", cluster_distances.shape, cluster_distances)
    #print("real distance", actualSetsDistances[cluster_elements[0], cluster_elements[0]])
    cluster_distances_sum = np.sum(cluster_distances, axis=1)
    cluster_distances_mean = np.mean(cluster_distances, axis=1)
    cluster_mean_distances.append(np.min(cluster_distances_mean))
    #print("cluster min mean distance: ", np.min(cluster_distances_mean))
    medoid = cluster_elements[np.argmin(cluster_distances_sum)]
    #medoidMean = cluster_elements[np.argmin(cluster_distances_mean)]
    #print("medoidMean", medoidMean, "medoid", medoid)
    #print("medoid", medoid)
    medoidsIndices.append(medoid)
medoidsIndices = np.array(medoidsIndices)

print("medoidsIndices: ", medoidsIndices.shape, medoidsIndices)
print("cluster_mean_distances: ", len(cluster_mean_distances), cluster_mean_distances)



##### t-SNE #####

matricesActualAndStandard = np.vstack([route_matrix, route_matrix_standard])
print("matricesActualAndStandard", matricesActualAndStandard.shape, matricesActualAndStandard[0])

perplexity = 30 if len(matricesActualAndStandard) > 30 else len(matricesActualAndStandard) - 1
completeSetTSNE = TSNE(n_components=3, perplexity=perplexity, n_iter=1000, verbose=1).fit_transform(matricesActualAndStandard)



##### FIND IMPROVEMENT/DECLINE #####

# reorder the labels to have colors matching the cluster results, using medoids which are closer to the standard vectors
medoidSets = [actualSets[i] for i in medoidsIndices]
print("medoidSets: ", len(medoidSets))
print("medoid indices: ", medoidsIndices.shape, medoidsIndices)

num_clusters_unique = unique_labels[unique_labels >= 0]
#print("num_clusters_unique: ", len(num_clusters_unique), num_clusters_unique)

assert len(medoidSets) == len(num_clusters_unique), "The number of medoids is not equal to the number of unique labels"   

if len(medoidSets) == 0:
    print("No clustroids found")
else:

    #route_matrix_standard, route_matrix_medoids = create_binary_matrices(standardSets, medoidSets)

    #simMatrixMixed = jaccard_similarity_two_matrices(route_matrix_medoids, route_matrix_standard)
    simMatrixMixed = route_similarity_standard_to_actual[medoidsIndices]
    print("simMatrixMixed: ", type(simMatrixMixed), simMatrixMixed.shape, simMatrixMixed[0])
    #print("route_matrix_standard: ", route_matrix_standard.shape)

    #print("route_matrix_actual: ", route_matrix_medoids.shape)

    #print("simMatrixMixed: ", simMatrixMixed.shape, simMatrixMixed[0])


    CAN_BE_ORDERED = False
    # get the closest standard vector for each medoid using simMatrixMixed

    # argmax = np.argmax(simMatrixMixed, axis=1) # get the index of the closest standard vector for each medoid
    # maxValues = np.max(simMatrixMixed, axis=1) # get the value of the closest standard vector for each medoid
    argmax = simMatrixMixed.argmax(axis=1) # get the index of the closest standard vector for each medoid
    maxValues = simMatrixMixed.max(axis=1).toarray() # get the value of the closest standard vector for each medoid
    #argmax = np.where(maxValues > 0.0, argmax, -1) # if the closest standard vector is not similar enough, set it to -1
    # argmax = np.asarray(argmax).ravel() #np.array([medoidsIndices[i] for i in argmax]) # map the index to the actual index in the completeSets
    # maxValues = np.asarray(maxValues).ravel()
    argmax = np.array(argmax).flatten()
    maxValues = maxValues.flatten()
    # maxValues = np.array(maxValues)
    #argmax = np.where(maxValues > 0.0, argmax, -1) # if the closest standard vector is not similar enough, set it to -1
    print("argmax: ", argmax.shape, type(argmax), argmax)
    print("maxval: ", maxValues.shape, type(maxValues), maxValues)
    

    if len(set(argmax)) == len(medoidsIndices): # if the argmax are all different, then the medoids can be reordered
        print("argmax is correct, can be reordered")
        CAN_BE_ORDERED = True
        # reorder medoidsIndices
        #print("argmax: ", argmax.shape, argmax)
        #print("medoidsIndices: ", medoidsIndices.shape, medoidsIndices)
        
        
        # Create an array of zeros with the same shape as the original array
        # result = np.zeros_like(medoidsIndices)
        # argsort = np.argsort(argmax)
        # print("argsort: ", argsort.shape, argsort)
        # # Fill the result array using the permutation indices
        # result[argsort] = medoidsIndices
        # medoidsIndicesReordered = result # reorder medoidsIndices
        # result = np.zeros_like(medoidsIndices)
        # unique_labels_filtered = unique_labels[unique_labels >= 0]
        # result[argsort] = unique_labels_filtered
        unique_labels_reordered = argmax   # 4, 2, 5, 10, ... -> 10, 4, 5, 2, ...
        
        #medoidsIndicesReordered = medoidsIndices[argmax] # reorder medoidsIndices
        #print("medoidsIndices: ", medoidsIndicesReordered.shape, medoidsIndicesReordered)
        #unique_labels = unique_labels[argmax]


    # for i in range(len(standardSets)):
    #     #distances = np.linalg.norm(medoids - standardVectors[i], axis=1)
    #     distancesCosine = []
    #     for j in range(len(medoidsIndices)):
    #         distancesCosine.append(cosine(actualSets[j], standardSets[i]))
    #     closest_medoid = np.argmin(distancesCosine)
        
    #     if closest_medoid not in unique_labels_reordered:
    #         unique_labels_reordered.append(closest_medoid)
    #     else:
    #         print("closest_medoid already in unique_labels_reordered")
    #         can_be_reordered = False
    #         break

    if not CAN_BE_ORDERED:
        #unique_labels_reordered = unique_labels

        #unique_labels = unique_labels_reordered
        #unique_labels = np.unique(labels_HDBSCAN)
        
        print("unique_labels: ", len(unique_labels), unique_labels)
    else:
        print("unique_labels_reordered: ", len(unique_labels_reordered), unique_labels_reordered)
        #unique_labels = unique_labels_reordered
        # compare if distances between clustroids and standard vectors are smaller than distances between standard vectors and other actual vectors

        #distancesClustroids = simMatrixMixed[np.arange(len(simMatrixMixed)), argmax]
        
    distancesClustroids = []
    distancesStandardVectors = []
    distancesStandardVectors2 = []
    # for i, clustroid in enumerate(medoidsIndices):
    #     distMedCluster = completeSetsDistances[:len(actualSets), :len(actualSets)][clustroid, labels_HDBSCAN == i]
    #     #print(labels_HDBSCAN)
    #     #print(labels_HDBSCAN == argmax[i])
    #     print("distMedCluster: ", np.sum(distMedCluster), len(distMedCluster), distMedCluster)
    #     # make distMedCluster a single list, no np.sum since they are inhomogeneous lists
    #     #print()
    #     #print(actualRefStandardIds)
    #     distStdCluster = completeSetsDistances[len(actualSets):, :len(actualSets)][i, np.where(np.array(actualRefStandardIds) == i)[0]]
    #     print("distStdCluster: ", np.sum(distStdCluster), len(distStdCluster), distStdCluster)
        
    #     distancesClustroids.append(np.mean(distMedCluster))
    #     distancesStandardVectors2.append(np.mean(distStdCluster))
    actualRefStandardIdsNumpy = np.array(actualRefStandardIds)
    for i, stdID in enumerate(standardRefIds):
        #distStdCluster = completeSetsDistances[len(actualSets):, :len(actualSets)][i, np.where(actualRefStandardIdsNumpy == stdID)[0]]
        #distSimCluster = completeSetsSimilarities[len(actualSets):, :len(actualSets)][i, np.where(actualRefStandardIdsNumpy == stdID)[0]]
        #print("stdID: ", stdID)
        #print("actualRefStandardIds", actualRefStandardIds)
        #print("where", len(np.where(actualRefStandardIdsNumpy == stdID)[0]), np.where(actualRefStandardIdsNumpy == stdID)[0])
        distSimCluster = route_similarity_standard_to_actual[np.where(actualRefStandardIdsNumpy == stdID)[0], i]
        distStdCluster = 1 - distSimCluster.toarray()
        distStdCluster = distStdCluster[distStdCluster != 1]
        print("distSimCluster: ", np.mean(distStdCluster), len(distStdCluster), distStdCluster)
        meanDist = np.mean(distStdCluster)
        if np.isnan(meanDist):
            distancesStandardVectors.append(1)
        else:
            distancesStandardVectors.append(np.mean(distStdCluster))
    
    
    mean_similarity_clustroids = np.mean(cluster_mean_distances)
    std_dev_similarity_clustroids = np.std(cluster_mean_distances)
    
    mean_similarity_standard_vectors = np.mean(distancesStandardVectors)
    std_dev_similarity_standard_vectors = np.std(distancesStandardVectors)
    
    cv_clustroids = std_dev_similarity_clustroids / mean_similarity_clustroids
    cv_standard_vectors = std_dev_similarity_standard_vectors / mean_similarity_standard_vectors
    
    # print in green if the improvement is positive, in red if it is negative
    print("\n\033[94mMean similarity from vectors of the same cluster to:")
    print("         clustroids: ", mean_similarity_clustroids)
    #print("         first     : ", np.mean(distancesClustroids))
    print("   standard vectors: ", mean_similarity_standard_vectors)
    #print("first                ", np.mean(distancesStandardVectors2))
    
    
    print("\nStd dev similarity from vectors of the same cluster to:")
    print("         clustroids: ", std_dev_similarity_clustroids)
    print("   standard vectors: ", std_dev_similarity_standard_vectors)
    
    print("\nCoefficient of variation from vectors of the same cluster to:")
    print("         clustroids: ", cv_clustroids)
    print("   standard vectors: ", cv_standard_vectors)
    print("\033[0m")
    
    #ratio = np.mean(distancesClustroids) / np.mean(distancesStandardVectors)
    ratio = mean_similarity_standard_vectors / mean_similarity_clustroids
    percentage = ratio * 100
    
    print("\033[93mMean:\033[0m")
    if percentage >= 100:
        # print in green if the improvement is positive, in red if it is negative
        print("   Improvement: \033[92m{:.2f}% \033[0m".format(percentage-100))
    else:
        print("   Decline: \033[91m{:.2f}% \033[0m".format(100-percentage))
        
    
    ratio = std_dev_similarity_standard_vectors / std_dev_similarity_clustroids
    percentage = ratio * 100
    print("\033[93m\nStd dev:\033[0m")
    if percentage >= 100:
        # print in green if the improvement is positive, in red if it is negative
        print("   Improvement: \033[92m{:.2f}% \033[0m".format(percentage-100))
    else:
        print("   Decline: \033[91m{:.2f}% \033[0m".format(100-percentage))
        
    ratio =  cv_standard_vectors / cv_clustroids
    percentage = ratio * 100
    print("\033[93m\nCoefficient of Variation:\033[0m")
    if percentage >= 100:
        # print in green if the improvement is positive, in red if it is negative
        print("   Improvement: \033[92m{:.2f}% \033[0m".format(percentage-100))
    else:
        print("   Decline: \033[91m{:.2f}% \033[0m\n".format(100-percentage))
        
        
        
        
        
        

######### PLOT CLUSTERS FOUND #########

max_len = max(len(standardSets), len(num_clusters_unique))
colors = plt.cm.jet(np.linspace(0, 1, max_len))
# print("colors", colors.shape, colors)

# print("medoid labels", [labels_HDBSCAN[i] for i in medoidsIndices])
# print("argmax", argmax)

if len(medoidSets) > 0 and CAN_BE_ORDERED:
    print("medoids added to plots and reordered")
    color_map = dict(zip(range(max_len), colors[unique_labels_reordered]))
    #rint("color_map", color_map)
else:
    color_map = dict(zip(range(max_len), colors))   # 0=red, 1=blue, 2=green, 3=yellow, 4=purple, 5=lightblue, 6=lightgreen, 7=lightyellow, 8=lightpurple
    
marker_colors = [color_map[label] if label > -1 else np.array([0,0,0,1]) for label in labels_HDBSCAN]
marker_colors_medoids = [color_map[label] if label > -1 else np.array([0,0,0,1]) for label in labels_HDBSCAN[medoidsIndices]]
#marker_colors_medoids = [color_map[label] if label > -1 else np.array([0,0,0,1]) for label in unique_labels]


# Create a trace for each type (centroids data)
traceStandard = go.Scatter3d(
    x=completeSetTSNE[len(actualSets):,0],
    y=completeSetTSNE[len(actualSets):,1],
    z=completeSetTSNE[len(actualSets):,2],
    mode='markers',
    marker=dict(
        size=7,
        color=colors,                # set color to an array/list of desired values
        opacity=1,
        symbol='diamond'
    )
)

if len(medoidSets) > 0:
    medoidsElements = completeSetTSNE[medoidsIndices] # medoidsIndices = [921, 123]
    traceMedoids = go.Scatter3d(
        x=medoidsElements[:,0],
        y=medoidsElements[:,1],
        z=medoidsElements[:,2],
        mode='markers',
        marker=dict(
            size=7,
            color=marker_colors_medoids,                # set color to an array/list of desired values
            opacity=1,
            symbol='cross'
        )
    )

# Create a trace for each type (centroids data)
traceActual = go.Scatter3d(
    x=completeSetTSNE[:len(actualSets),0],
    y=completeSetTSNE[:len(actualSets),1],
    z=completeSetTSNE[:len(actualSets),2],
    mode='markers',
    marker=dict(
        size=7,
        color=marker_colors,                # set color to an array/list of desired values
        opacity=0.1,
        symbol='circle'
    )
)

# Plot
if len(medoidSets) > 0:
    data = [traceStandard, traceActual, traceMedoids]
else:
    data = [traceStandard, traceActual]

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

fig = go.Figure(data=data, layout=layout)
fig.show()



######## PLOT GROUND TRUTH ########


colors_true = plt.cm.jet(np.linspace(0, 1, len(standardSets)))
color_map_true = dict(zip(range(max_len), colors))   # 0=red, 1=blue, 2=green, 3=yellow, 4=purple, 5=lightblue, 6=lightgreen, 7=lightyellow, 8=lightpurple
# marker colors for each point with the same color as the cluster it belongs to in the original data
marker_colors_true = [color_map_true[label] for label in actualRefStandardIds]


# Create a trace for each type (centroids data)
traceStandard_true = go.Scatter3d(
    x=completeSetTSNE[len(actualSets):,0],
    y=completeSetTSNE[len(actualSets):,1],
    z=completeSetTSNE[len(actualSets):,2],
    mode='markers',
    marker=dict(
        size=7,
        color=colors_true,                # set color to an array/list of desired values
        opacity=1,
        symbol='diamond'
    )
)

# Create a trace for each type (centroids data)
traceActual_true = go.Scatter3d(
    x=completeSetTSNE[:len(actualSets),0],
    y=completeSetTSNE[:len(actualSets),1],
    z=completeSetTSNE[:len(actualSets),2],
    mode='markers',
    marker=dict(
        size=7,
        color=marker_colors_true,                # set color to an array/list of desired values
        opacity=0.1,
        symbol='circle'
    )
)

# Plot
data = [traceStandard_true, traceActual_true]

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

fig = go.Figure(data=data, layout=layout)
fig.show()


##### OUTPUT RECOMMENDED STANDARD ROUTES #####

if not os.path.exists(os.path.join(HOME, "results")):
    os.makedirs(os.path.join(HOME, "results"))
    
# Save the medoids to a file
with open(os.path.join(HOME, "results", 'recStandard.json'), 'w', encoding="utf-8") as f:
    recStandard = []
    for i, index in enumerate(medoidsIndices):
        recRoute = {"id": "s" + str(i)}
        recRoute["route"] = dfActual.iloc[actualSets[map_indices_back[index]][0]]["route"]
        recStandard.append(recRoute)
    json.dump(recStandard, f, ensure_ascii=False, indent=2)

print("\n\nRecommended standard routes saved to file 'results/recStandard.json'")
    
    
print("TASK 1 FINISHED")








###########################
#                         #
#          TASK 2         #
#                         #
###########################

print("\nroute_matrix_distance_actual_standard", route_similarity_standard_to_actual[99:101])

max_value = np.max(route_similarity_standard_to_actual, axis=1).toarray()
print("max_value", max_value.shape, max_value[0:12])
max_value_index = np.argmax(route_similarity_standard_to_actual, axis=1)
print("max_value_index", max_value_index.shape, max_value_index[0:12])
max_value_index[max_value == 0] = -1

print("max_value_index", max_value_index.shape, max_value_index[0:12])


print("len(max_value_index)", len(max_value_index))
print("len(max_value)", len(max_value))

# [index for index, s in dfActual.iterrows() if s['driver'] == driver]
# uniqueDrivers
driver_indices = {}

for i, s in dfActual.iterrows():
    driver = s['driver']
    
    # Check if the driver is already in the dictionary
    if driver in driver_indices:
        # If yes, append the index to the existing array
        driver_indices[driver].append(i)
    else:
        # If not, create a new array with the current index
        driver_indices[driver] = [i]

print("driver_indices", driver_indices)
print("len(driver_indices)", len(driver_indices))



# Create a dictionary of all drivers' routes
drivers_routes = {}

for driver in uniqueDrivers:
    print("driver", driver)

    driver_standard_index = np.array(max_value_index[driver_indices[driver]])
    driver_max_value = np.array(max_value[driver_indices[driver]])

    # Assuming driver_standard_index is a NumPy array
    unique_values_index = np.unique(driver_standard_index[~np.isnan(driver_standard_index)]).astype(int)
    # print("unique_values_index", unique_values_index)

    # Calculate the mean for each unique value
    # means = [np.mean(driver_max_value[driver_standard_index == idx]) for idx in unique_values_index if idx != np.nan]
    # print("means", means)

    weighted_sums = [np.sum(driver_max_value[driver_standard_index == idx]) * np.count_nonzero(driver_standard_index == idx) for idx in unique_values_index if not np.isnan(idx)]
    # print("weighted_sums", weighted_sums)

    best_route_Ids = []
    # Print the results for each driver
    for idx, mean in zip(unique_values_index, weighted_sums):
        # print(f"Driver: {driver}, Unique Value: {idx}, Mean: {mean}")
        best_route_Ids.append([standardIds[idx], mean])

    # Sort the routes by their mean
    best_route_Ids.sort(key=lambda x: x[1], reverse=True)
    # print("best_route_Ids", best_route_Ids)
    
    # Keep the top 5 routes
    top_5_routes = best_route_Ids[:5]

    # Update the driver's routes in the dictionary
    drivers_routes[driver] = {'driver': driver, 'routes': [id for id,value in top_5_routes]}

# Convert the dictionary to a list for JSON serialization
result_list = list(drivers_routes.values())

# Write the result to driver.json
with open(os.path.join(HOME, 'results', 'driver.json'), 'w', encoding="utf-8") as outfile:
    json.dump(result_list, outfile, ensure_ascii=False ,indent=2)

print("JSON driver data has been written to results/driver.json")




###########################
#                         #
#          TASK 3         #
#                         #
###########################


# Create a dictionary of all drivers' routes

## Dictionary containing all the Sets [idx, [shingles], [quantity-hot merch]] for all the drivers
# actualDriverSets = {}
# for driver in uniqueDrivers:
#     dfActualDriver = dfActual[dfActual['driver'] == driver]
#     actualDriverSets[driver] = createShingles(dfActualDriver, k=K_SHINGLES, uniqueCities=uniqueCities, uniqueItems=uniqueItems, longestRoute=longestRoute, maxItemQuantity=maxItemQuantity, permutations=permutations)


# def mapping_back(indices, mapping):
#     new_indices = []
#     for key, value in mapping.items():
#         new_indices.append(indices[value])
#     return new_indices
        

## Dictionary containing the distance matrix for all the drivers
driversDistances = {}
driversIndicesMapped = {}
driversIndicesBack = {}
for driver in uniqueDrivers:
    
    #driver_indices = [i[0] for i in driverSet]
    #route_matrix = create_binary_matrix(driverSet)
    #route_matrix_driver = route_matrix[driver_indices[driver]]
    print("driver indices", driver_indices[driver])

    # num_hash_functions = find_num_hashes_minhash(route_matrix)
    # route_matrix = minhash(route_matrix, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)

    # merch_matrix = np.array([s[2] for s in driverSet])

    #actualSetsDistances, mapping= jaccard_similarity_minhash_lsh_route_merch(route_matrix, merch_matrix, thresh_user=0.2)

    #driver_indices = mapping_back(driver_indices[driver], mapping)
    driver_indices_mapped =  [ map_indices[i] for i in driver_indices[driver] if i in map_indices ]
    driver_indices_mapped_back =  [ map_indices_back[i] for i in driver_indices_mapped ]
    print('driver_indices_mapped', driver_indices_mapped)
    print('driver_indices_mapped_back', driver_indices_mapped_back)

    #driversDistances[driver] = actualSetsDistances
    driversIndicesMapped[driver] = driver_indices_mapped
    driversIndicesBack[driver] = driver_indices_mapped_back
    
    

from sklearn.metrics import pairwise_distances

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

    return np.mean(mean_distances)

def intra_cluster_distance(sparse_distances):
    mean_distances = []
    for point in sparse_distances:
        mean_distances.append(np.mean(point))

    return mean_distances

best_clustroids = []

driversClustersPoints = {}
driversLabels = {}
driversClusterLabels = {}
driversIdealPoint = {}
driversIdealIndex = {}

for driver, driverIndices in driversIndicesMapped.items():
    print('performing DBSCAN for driver ', driver)

    #print(f'best min {best_min} best max {best_max}')
    #print('driverIndices', driverIndices)
    driverDistance = actualSetsDistances[driverIndices]
    #print('driverDistance', type(driverDistance), driverDistance.shape, driverDistance[0])
    driverDistance = csr_matrix(driverDistance[:, driverIndices])
    #print('driverDistance', type(driverDistance), driverDistance.shape, driverDistance[0])
    hdb = HDBSCAN(min_cluster_size=2, max_cluster_size=forward_expansion, metric="precomputed", store_centers=None, allow_single_cluster=False ).fit(driverDistance.copy())
    print(f"num clusters found for driver {driver}: {len(set(hdb.labels_))}")
    print('labels: ', hdb.labels_)
    labels = hdb.labels_

    ## Selecting the cluster that has the higher intra-cluster similarity
    # Compute pairwise similarity/distance matrix within clusters
    cluster_distances = []
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_points = driverDistance[labels == label]
        cluster_distance = inter_cluster_distance(cluster_points)
        cluster_distances.append(cluster_distance)

    # Select the cluster with the lowest distance
    selected_cluster = np.argmin(cluster_distances)
    #print('selected cluster',selected_cluster)
    # Access the data points in the selected cluster
    selected_cluster_points = driverDistance[labels == selected_cluster]
    selected_cluster_distance = intra_cluster_distance(selected_cluster_points)
    # Find the medoid that maximize the inter-cluster similarity
    selected_point = np.argmin(selected_cluster_distance)
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

data = []
for driver in uniqueDrivers:
    data.append({'driver':driver, 'route':dfActual.loc[driversIdealIndex[driver]]['route']})

# Write the data to the JSON file
file_path = os.path.join(HOME, 'results', "perfectRoute.json") #"standard.json"
with open(file_path, 'w', encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=2, ensure_ascii=False)
    
    
columns = 2
rows = len(uniqueDrivers) // columns
fig, axs = plt.subplots(rows, columns, figsize = (columns*3, rows*3))

axs = axs.flatten()

for i, driver in enumerate(uniqueDrivers):
    idx = best_clustroids[i]

    fit = umap.UMAP(n_neighbors=15,min_dist=0.1,n_components=2,metric='cosine')
    driverDistance = actualSetsDistances[driversIndicesMapped[driver]]
    driverDistance = csr_matrix(driverDistance[:, driversIndicesMapped[driver]])
    u = fit.fit_transform(driverDistance)

    axs[i].scatter(u[:,0], u[:,1], s = 50, c=driversLabels[driver])

    axs[i].plot(u[idx, 0], u[idx, 1], marker='*', markersize=5, color='black')

    axs[i].axhline(y=u[idx, 1], color='gray', linestyle='--')
    axs[i].axvline(x=u[idx, 0], color='gray', linestyle='--')

    axs[i].set_title(f"driver {driver}", fontsize=12)
    #axs[i].legend()

plt.tight_layout()
plt.show()



columns = 2
rows = len(uniqueDrivers) // columns
fig = plt.figure(figsize=(columns * 3, rows * 3))

for i, driver in enumerate(uniqueDrivers):
    idx = best_clustroids[i]

    fit = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, metric='euclidean')
    driverDistance = actualSetsDistances[driversIndicesMapped[driver]]
    driverDistance = csr_matrix(driverDistance[:, driversIndicesMapped[driver]])
    u = fit.fit_transform(driverDistance)

    ax = fig.add_subplot(rows, columns, i + 1, projection='3d')
    ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=50, c=driversLabels[driver])

    ax.plot(u[idx, 0], u[idx, 1], u[idx, 2], marker='*', markersize=12, color='black')

    ax.set_title(f"driver {driver}", fontsize=12)

plt.tight_layout()
plt.show()