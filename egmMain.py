import os
import sys

import json
import pandas as pd
import numpy as np
import time

from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN

import matplotlib.pyplot as plt
import umap
import plotly.graph_objects as go

from scipy.sparse import csr_matrix

from tqdm import tqdm

import argparse


from utils.shingles import *
from utils.clustering import *


# Create the parser
parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument('-s', '--STANDARD_FILE', type=str, required=True, help='The standard routes file')
parser.add_argument('-a', '--ACTUAL_FILE', type=str, required=True, help='The actual routes file')
parser.add_argument('-p', '--PLOT', default=False, action='store_true', help='Whether to plot the results or not')
parser.add_argument('-n', '--NUM_PLOTS', type=int, default=4, help='Number of plots to show')
parser.add_argument('-k', '--K_SHINGLES', type=int, default=3, help='Number of k-shingles to use')
parser.add_argument('-m', '--METRIC', type=str, default="jaccard", choices=["cosine", "jaccard", "L2"], help='Metric to use for the similarity between routes')
parser.add_argument('-f', '--FUSION', type=str, default="mean", choices=["mean", "product", "weighted product"], help='Fusion method to use for route and merchandise similarity')
parser.add_argument('--alpha', type=float, default=0.5, help='Weight of the route vector wrt merchandise vector, between 0 and 1')


args = parser.parse_args()

# Check if NUM_PLOTS is set and PLOT is not
if args.NUM_PLOTS != 4 and not args.PLOT:
    parser.error("--PLOT is required when --NUM_PLOTS is set")
    
if args.alpha < 0 or args.alpha > 1:
    parser.error("--ALPHA must be between 0 and 1")
    
if args.K_SHINGLES < 2:
    parser.error("--K_SHINGLES must be greater than 1")

HOME = os.path.dirname(os.path.realpath(__file__))
print('HOME: ',HOME)

sys.path.append(HOME)


### GLOBAL VARIABLES ###

STANDARD_FILE = args.STANDARD_FILE
ACTUAL_FILE = args.ACTUAL_FILE

STANDARD_FILE_NAME = os.path.basename(STANDARD_FILE)
ACTUAL_FILE_NAME = os.path.basename(ACTUAL_FILE)

K_SHINGLES = args.K_SHINGLES

PLOT = args.PLOT
NUM_PLOTS = args.NUM_PLOTS

ALPHA = args.alpha    # weight of route vector wrt merchandise vector, between 0 and 1
METRIC = args.METRIC  # "cosine" or "jaccard" or "L2", metric used for merchandise vectors
FUSION = args.FUSION     # "mean" or "product" or "weigthed product", how to fuse the route and merchandise vectors 
                    # (e.g. mean: (ALPHA) * route + (1-ALPHA) * merchandise, product: route * merchandise, weighted product: route * (merchandise + 1))




#### READ DATA ####

time_start = time.time()

assert STANDARD_FILE_NAME.startswith('standard') and ACTUAL_FILE_NAME.startswith('actual'), "The files must start with 'standard' and 'actual'"

print("\nParamters used:")
print("STANDARD_FILE: ", STANDARD_FILE)
print("ACTUAL_FILE: ", ACTUAL_FILE)
print("K_SHINGLES: ", K_SHINGLES)
print("PLOT: ", PLOT)
print("NUM_PLOTS: ", NUM_PLOTS)
print("METRIC: ", METRIC)
print("FUSION: ", FUSION)
print("ALPHA: ", ALPHA)

# load standard and actual data
print("\nReading standard data...")
with open(os.path.join(STANDARD_FILE), encoding="utf-8") as f:
    standard = json.load(f)
    
SUFFIX_FILE = STANDARD_FILE.split('.')[0].split('standard')[-1]


print("\nReading actual data...")
with open(os.path.join(ACTUAL_FILE), encoding="utf-8") as f:
    actual = json.load(f)


# load the data into a dataframe
print("\nCreating standard dataframe...")
dfStandard = pd.DataFrame(standard)
print("\nCreating actual dataframe...")
dfActual = pd.DataFrame(actual)

# get the unique cities and items of the standard data
cities = []
items = []
drivers = []
longestRoute = 0
shortestRoute = np.inf
maxItemQuantity = 0

standardRefIds = []
for index, s in dfStandard.iterrows():
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
dictShingles = {}
for index, s in dfActual.iterrows():
    idS = s['id']
    route = s['route']
    idStandard = s['sroute']
    actualRefStandardIds.append(int(idStandard[1]))
    drivers.append(s['driver'])
    current_cities = []
    for trip in route:
        cities.append(trip['from'])
        current_cities.append(trip['from'])
        items.extend(trip['merchandise'].keys())
        maxItemQuantity = max(maxItemQuantity, max(trip['merchandise'].values()))
        
    if len(route) > 0:
        cities.append(route[-1]['to'])
        current_cities.append(route[-1]['to'])
        
    if len(route) > longestRoute:
        longestRoute = len(route)
    
    if len(route) < shortestRoute:
        shortestRoute = len(route)
    
    if K_SHINGLES >= 3 and shortestRoute >= K_SHINGLES - 1:
        for i in range(len(current_cities)-K_SHINGLES+1):
            shingle = hash_shingles(current_cities[i:i+K_SHINGLES])
            if shingle not in dictShingles:
                dictShingles[shingle] = 1
        
print("\nFinished preparing actual data")


# find the unique cities and items
uniqueCities = sorted(list(set(cities)))
#uniqueCities.insert(0, 'NULL')          # add NULL city, for padding vectors with different lengths (trips in routes)
uniqueItems = sorted(list(set(items)))
uniqueDrivers = sorted(list(set(drivers)))

print("\nSorted cities and items")

if shortestRoute < 2 or len(dictShingles) * len(dfActual) > 100_000_000:
    print("\n\033[91mK-Shingles too big, setting to 2\033[0m")
    K_SHINGLES = 2

standardIds = dfStandard['id'].tolist()

print("\nNumber of cities: ", len(uniqueCities))
print("Number of items: ", len(uniqueItems))

print("\nLongest route: ", longestRoute)
print("Shortest route: ", shortestRoute)

print("\nMax item quantity: ", maxItemQuantity)

print(f"\n\033[92mK-Shingles used: {K_SHINGLES} \033[0m")



#############################################




##### CREATE SHINGLES #####

print("\nCreating shingles...")
#pandarallel.initialize(progress_bar=True)
standardSets = createShingles(dfStandard, K_SHINGLES, uniqueCities, uniqueItems, longestRoute, maxItemQuantity)
#standardSets = standardSets.tolist()
actualSets = createShingles(dfActual, K_SHINGLES, uniqueCities, uniqueItems, longestRoute, maxItemQuantity)
#actualSets = actualSets.tolist()

assert len(standardSets[0]) == 3, "The length of the standard set is not equal to 3 (index, shingles, merchandise)"
assert len(standardSets[0][2]) == len(uniqueItems), "The length of the merchandise vector is not equal to the number of unique items"


######## ESSENTIALS ########

# convert routes and merchandise to binary matrices
# binary matrix where each row represents a route
print("Creating route binary matrix...")
route_matrix, route_matrix_standard = create_binary_matrices(actualSets, standardSets)
print("Route matrix shape: ", route_matrix.shape)

print("Minhashing route and standard matrix...")    
num_hash_functions = find_num_hashes_minhash(route_matrix)
route_matrix, route_matrix_standard = minhash_matrices(route_matrix, route_matrix_standard, num_hash_functions if num_hash_functions % 2 == 0 else num_hash_functions + 1)
print("Route minhashed matrix shape: ", route_matrix.shape)

print("Creating merchandise binary matrix...")
merch_matrix = np.array([s[2] for s in actualSets])


print("Computing Jaccard similarity route matrix...")
threshold_lsh = find_threshold_lsh(route_matrix, route_matrix)
actualSetsDistances, map_indices, map_indices_back = jaccard_similarity_minhash_lsh_route_merch(route_matrix, merch_matrix, thresh_user=threshold_lsh, metric=METRIC, fusion=FUSION, alpha=ALPHA)
print("Distance matrix shape: ", actualSetsDistances.shape)


####### ESSENTIALS FOR TASK 2 ########
print("\n\nTASK 2 ESSENTIALS\n\n")

merch_matrix_standard = np.array([s[2] for s in standardSets])
 
threshold_lsh_actual_to_standard = find_threshold_lsh(route_matrix, route_matrix_standard)
print("Computing Jaccard similarity actual to standard route matrix...")
route_similarity_standard_to_actual = similarity_minhash_lsh_two_matrices_and_merch(route_matrix, merch_matrix, route_matrix_standard, merch_matrix_standard, thresh_user=threshold_lsh_actual_to_standard, metric=METRIC, fusion=FUSION, alpha=ALPHA)
print("Similarity matrix actual-to-standard shape: ", route_similarity_standard_to_actual.shape)

###### CLUSTERING ######

# HDBSCAN clustering

# compute mean forward expansion
forward_expansion = len(actualSets) // len(standardSets)


print("Computing HDBSCAN...")
hdb = HDBSCAN(min_cluster_size=forward_expansion//3, max_cluster_size=forward_expansion, metric="precomputed", store_centers=None,allow_single_cluster=False).fit(actualSetsDistances.copy())

labels_HDBSCAN = hdb.labels_
print("Number of clusters found:", len(set(labels_HDBSCAN)))
print("Biggest cluster:", max(labels_HDBSCAN, key=list(labels_HDBSCAN).count), " num elements: ", list(labels_HDBSCAN).count(max(labels_HDBSCAN, key=list(labels_HDBSCAN).count)))

# Create a color map that maps each unique label to a color
unique_labels = np.unique(labels_HDBSCAN)
#unique_labels = unique_labels[unique_labels != -1]
print("Clusters: ", unique_labels)
print("Number standard routes:", len(standardSets))

# find the medoids using the clusters found by HDBSCAN
medoidsIndices = []
cluster_mean_distances = []
for cluster in unique_labels:
    if cluster in [-1, -2, -3]:
        continue
    cluster_elements = np.where(labels_HDBSCAN == cluster)[0]
    cluster_distances = actualSetsDistances[cluster_elements][:,cluster_elements]
    cluster_distances_sum = np.sum(cluster_distances, axis=1)
    cluster_distances_mean = np.mean(cluster_distances, axis=1)
    cluster_mean_distances.append(np.min(cluster_distances_mean))
    medoid = cluster_elements[np.argmin(cluster_distances_sum)]
    medoidsIndices.append(medoid)
medoidsIndices = np.array(medoidsIndices)

##### t-SNE #####

if PLOT:
    matricesActualAndStandard = np.vstack([route_matrix, route_matrix_standard])

    perplexity = 30 if len(matricesActualAndStandard) > 30 else len(matricesActualAndStandard) - 1
    completeSetTSNE = TSNE(n_components=3, perplexity=perplexity, n_iter=1000, verbose=1).fit_transform(matricesActualAndStandard)



##### FIND IMPROVEMENT/DECLINE #####

# reorder the labels to have colors matching the cluster results, using medoids which are closer to the standard vectors
medoidSets = [actualSets[i] for i in medoidsIndices]

num_clusters_unique = unique_labels[unique_labels >= 0]

assert len(medoidSets) == len(num_clusters_unique), "The number of medoids is not equal to the number of unique labels"   

if len(medoidSets) == 0:
    print("No clustroids found, cannot assign closest standard vector to each medoid")
else:
    simMatrixMixed = route_similarity_standard_to_actual[medoidsIndices]

    CAN_BE_ORDERED = False
    # get the closest standard vector for each medoid using simMatrixMixed
    argmax = simMatrixMixed.argmax(axis=1) # get the index of the closest standard vector for each medoid
    maxValues = simMatrixMixed.max(axis=1).toarray() # get the value of the closest standard vector for each medoid
    argmax = np.array(argmax).flatten()
    maxValues = maxValues.flatten()
    print("Closest standard routes for medoids: ", argmax.shape, type(argmax), argmax)

    if len(set(argmax)) == len(medoidsIndices): # if the argmax are all different, then the medoids can be reordered
        print("Medoids have different closest standard route, can be reordered")
        CAN_BE_ORDERED = True

        unique_labels_reordered = argmax 
    
    distancesStandardVectors = []

    actualRefStandardIdsNumpy = np.array(actualRefStandardIds)
    for i, stdID in enumerate(standardRefIds):
        distSimCluster = route_similarity_standard_to_actual[np.where(actualRefStandardIdsNumpy == stdID)[0], i]
        distStdCluster = 1 - distSimCluster.toarray()
        distStdCluster = distStdCluster[distStdCluster != 1]
        if distStdCluster.shape[0] == 0:
            distancesStandardVectors.append(1)
        else:
            distancesStandardVectors.append(np.mean(distStdCluster))
    
    
    mean_distance_clustroids = np.mean(cluster_mean_distances)
    std_dev_distance_clustroids = np.std(cluster_mean_distances)
    
    mean_distance_standard_vectors = np.mean(distancesStandardVectors)
    std_dev_distance_standard_vectors = np.std(distancesStandardVectors)
    
    cv_clustroids = std_dev_distance_clustroids / mean_distance_clustroids
    cv_standard_vectors = std_dev_distance_standard_vectors / mean_distance_standard_vectors
    
    # print in green if the improvement is positive, in red if it is negative
    print("TASK 1 RESULTS:")
    print("\n\033[94mMean distance from vectors of the same cluster to:")
    print("         clustroids: ", mean_distance_clustroids)
    print("   standard vectors: ", mean_distance_standard_vectors)
    
    
    print("\nStd dev distance from vectors of the same cluster to:")
    print("         clustroids: ", std_dev_distance_clustroids)
    print("   standard vectors: ", std_dev_distance_standard_vectors)
    
    print("\nCoefficient of variation from vectors of the same cluster to:")
    print("         clustroids: ", cv_clustroids)
    print("   standard vectors: ", cv_standard_vectors)
    print("\033[0m")
    
    ratio = mean_distance_standard_vectors / mean_distance_clustroids
    percentage = ratio * 100
    
    print("\033[93mMean:\033[0m")
    if percentage >= 100:
        # print in green if the improvement is positive, in red if it is negative
        print("   Improvement: \033[92m{:.2f}% \033[0m".format(percentage-100))
    else:
        print("   Decline: \033[91m{:.2f}% \033[0m".format(100-percentage))
        
    
    ratio = std_dev_distance_standard_vectors / std_dev_distance_clustroids
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

if PLOT:
    max_len = max(len(standardSets), len(num_clusters_unique))
    colors = plt.cm.jet(np.linspace(0, 1, max_len))

    if len(medoidSets) > 0 and CAN_BE_ORDERED:
        print("Medoids added to the plot, and reordered to match the closest standard routes")
        color_map = dict(zip(range(max_len), colors[unique_labels_reordered]))
    else:
        color_map = dict(zip(range(max_len), colors)) 
        
    marker_colors = [color_map[label] if label > -1 else np.array([0,0,0,1]) for label in labels_HDBSCAN]
    if len(medoidSets) > 0:
        marker_colors_medoids = [color_map[label] if label > -1 else np.array([0,0,0,1]) for label in labels_HDBSCAN[medoidsIndices]]


    # Create a trace for each type (centroids data)
    traceStandard = go.Scatter3d(
        x=completeSetTSNE[len(actualSets):,0],
        y=completeSetTSNE[len(actualSets):,1],
        z=completeSetTSNE[len(actualSets):,2],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,                # set color to an array/list of desired values
            opacity=1,
            symbol='diamond'
        ),
        name="Standard routes"
    )

    if len(medoidSets) > 0:
        medoidsElements = completeSetTSNE[[map_indices_back[i] for i in medoidsIndices]] 
        traceMedoids = go.Scatter3d(
            x=medoidsElements[:,0],
            y=medoidsElements[:,1],
            z=medoidsElements[:,2],
            mode='markers',
            marker=dict(
                size=10,
                color=marker_colors_medoids, # set color to an array/list of desired values
                opacity=1,
                symbol='cross'
            ),
            name="Recommended standard routes"  
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
        ),
        name="Actual routes"
    )

    # Plot
    if len(medoidSets) > 0:
        data = [traceStandard, traceActual, traceMedoids]
    else:
        data = [traceStandard, traceActual]

    layout = go.Layout(
        title="t-SNE of Actual routes and Standard routes, with HDBSCAN clusters",
        title_x=0.5,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=30
        ),
        legend=dict(
            orientation="v",
            yanchor="auto",
            y=1,
            xanchor="right",  # changed
            x=-0.3
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()



    ######## PLOT GROUND TRUTH ########


    colors_true = plt.cm.jet(np.linspace(0, 1, len(standardSets)))
    color_map_true = dict(zip(range(len(standardSets)), colors_true))   # 0=red, 1=blue, 2=green, 3=yellow, 4=purple, 5=lightblue, 6=lightgreen, 7=lightyellow, 8=lightpurple
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
        ),
        name="Standard routes"
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
        ),
        name="Actual routes"
    )

    # Plot
    data = [traceStandard_true, traceActual_true]

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=30
        ),
        legend=dict(
            orientation="v",
            yanchor="auto",
            y=1,
            xanchor="right",  # changed
            x=-0.3
        ),
        title="t-SNE of Actual routes and Standard routes, with Ground Truth clusters",
        title_x=0.5,
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()


##### OUTPUT RECOMMENDED STANDARD ROUTES #####


if not os.path.exists(os.path.join(HOME, "results")):
    print("Creating results directory...")
    os.makedirs(os.path.join(HOME, "results"))
    
# Save the medoids to a file
print("\nSaving recommended standard routes to file...")
with open(os.path.join(HOME, "results", 'recStandard' + SUFFIX_FILE + '.json'), 'w', encoding="utf-8") as f:
    recStandard = []
    for i, index in enumerate(medoidsIndices):
        recRoute = {"id": "s" + str(i)}
        recRoute["route"] = dfActual.iloc[actualSets[map_indices_back[index]][0]]["route"]
        recStandard.append(recRoute)
    json.dump(recStandard, f, ensure_ascii=False, indent=2)

print("\n\nRecommended standard routes saved to file 'results/recStandard" + SUFFIX_FILE + ".json'")
    
    
print("\nTASK 1 FINISHED\n")








###########################
#                         #
#          TASK 2         #
#                         #
###########################

print("Finding the top 5 standard routes for each driver...")
max_value = np.max(route_similarity_standard_to_actual, axis=1).toarray()
max_value_index = np.argmax(route_similarity_standard_to_actual, axis=1)
max_value_index[max_value == 0] = -1

# uniqueDrivers
driver_indices = {}
driver_std_id = {}

for i, s in dfActual.iterrows():
    driver = s['driver']
    standard_ruote_id = s['sroute']
    
    # Check if the driver is already in the dictionary
    if driver in driver_indices:
        # If yes, append the index to the existing array
        driver_indices[driver].append(i)
        driver_std_id[driver].append(standard_ruote_id)
    else:
        # If not, create a new array with the current index
        driver_indices[driver] = [i]
        driver_std_id[driver] = [standard_ruote_id]



# Create a dictionary of all drivers' routes
drivers_routes = {}
mean_top1 = []
mean_top2 = []
mean_top3 = []
mean_top4 = []
mean_top5 = []

for driver in uniqueDrivers:

    driver_standard_index = np.array(max_value_index[driver_indices[driver]])
    driver_max_value = np.array(max_value[driver_indices[driver]])

    # Calculate the mean for each unique value
    weighted_sums = [ np.mean(driver_max_value[driver_standard_index == idx]) * np.count_nonzero(driver_standard_index == idx) for idx in range(len(standardIds)) if idx != -1]
    weighted_sums = np.nan_to_num(weighted_sums)
    # print("weighted sums",weighted_sums)   
    count_value = [np.count_nonzero(driver_standard_index == idx) for idx in range(len(standardIds)) if idx!=-1]
    # print("count value",count_value)
    
    best_route_Ids = []
    # Print the results for each driver
    for idx, mean in zip(standardIds, weighted_sums):
        best_route_Ids.append([idx, mean])

    # Sort the routes by their mean
    best_route_Ids.sort(key=lambda x: x[1], reverse=True)

    # Keep the top 5 routes
    top_5_routes = best_route_Ids[:5]
    # print("top 5 routes",top_5_routes)

    # Update the driver's routes in the dictionary
    drivers_routes[driver] = {'driver': driver, 'routes': [id for id,value in top_5_routes]}
    
    mean_for_all_columns =  np.mean(route_similarity_standard_to_actual[driver_indices[driver]], axis=0)
    # Multiply the mean of all columns by the counts
    score_for_all_columns = np.multiply(mean_for_all_columns, count_value)
    mean_score_driver = np.mean(score_for_all_columns)
    
    # Calculate the mean for each top 5 routes
    mean_score_over_top1 = mean_score_driver / best_route_Ids[0][1] if best_route_Ids[0][1] != 0 else 0
    mean_top1.append(mean_score_over_top1)
    mean_score_over_top2 = mean_score_driver / best_route_Ids[1][1] if best_route_Ids[1][1] != 0 else 0
    mean_top2.append(mean_score_over_top2)
    mean_score_over_top3 = mean_score_driver / best_route_Ids[2][1] if best_route_Ids[2][1] != 0 else 0
    mean_top3.append(mean_score_over_top3)
    mean_score_over_top4 = mean_score_driver / best_route_Ids[3][1] if best_route_Ids[3][1] != 0 else 0
    mean_top4.append(mean_score_over_top4)
    mean_score_over_top5 = mean_score_driver / best_route_Ids[4][1] if best_route_Ids[4][1] != 0 else 0
    mean_top5.append(mean_score_over_top5)



# Convert the dictionary to a list for JSON serialization
result_list = list(drivers_routes.values())

# Write the result to driver.json
print("\nWriting JSON driver data to results/driver" + SUFFIX_FILE + ".json...")
with open(os.path.join(HOME, 'results', 'driver' + SUFFIX_FILE + '.json'), 'w', encoding="utf-8") as outfile:
    json.dump(result_list, outfile, ensure_ascii=False ,indent=2)

print("JSON driver data has been written to results/driver.json")

print("\nTASK 2 RESULTS:")
print("\033[93m\nMean Drivers TOP5 Improvements:\033[0m")

if (1-np.mean(mean_top1)) < 0:
    print("   Mean Top1 Decline: \033[91m{:.2f}% \033[0m".format(-(1-np.mean(mean_top1)) * 100))
else:
    print("   Mean Top1 Improvement: \033[92m{:.2f}% \033[0m".format((1-np.mean(mean_top1)) * 100))

if (1-np.mean(mean_top2)) < 0:
    print("   Mean Top2 Decline: \033[91m{:.2f}% \033[0m".format(-(1-np.mean(mean_top2)) * 100))
else:
    print("   Mean Top2 Improvement: \033[92m{:.2f}% \033[0m".format((1-np.mean(mean_top2)) * 100))

if (1-np.mean(mean_top3)) < 0:
    print("   Mean Top3 Decline: \033[91m{:.2f}% \033[0m".format(-(1-np.mean(mean_top3)) * 100))
else:
    print("   Mean Top3 Improvement: \033[92m{:.2f}% \033[0m".format((1-np.mean(mean_top3)) * 100))

if (1-np.mean(mean_top4)) < 0:
    print("   Mean Top4 Decline: \033[91m{:.2f}% \033[0m".format(-(1-np.mean(mean_top4)) * 100))
else:
    print("   Mean Top4 Improvement: \033[92m{:.2f}% \033[0m".format((1-np.mean(mean_top4)) * 100))

if (1-np.mean(mean_top5)) < 0:
    print("   Mean Top5 Decline: \033[91m{:.2f}% \033[0m".format(-(1-np.mean(mean_top5)) * 100))
else:
    print("   Mean Top5 Improvement: \033[92m{:.2f}% \033[0m".format((1-np.mean(mean_top5)) * 100))


print("\nTASK 2 FINISHED\n")




###########################
#                         #
#          TASK 3         #
#                         #
###########################


## Dictionary containing the distance matrix for all the drivers
driversDistances = {}
driversIndicesMapped = {}
driversIndicesBack = {}
for driver in uniqueDrivers:
    driver_indices_mapped =  [ map_indices[i] for i in driver_indices[driver] if i in map_indices ]
    driver_indices_mapped_back =  [ map_indices_back[i] for i in driver_indices_mapped ]
    driversIndicesMapped[driver] = driver_indices_mapped
    driversIndicesBack[driver] = driver_indices_mapped_back

best_clustroids = []

driversClustersPoints = {}
driversLabels = {}
driversClusterLabels = {}
driversIdealPoint = {}
driversIdealIndex = {}

driver_best_divergence = {}
driver_cluster_divergences = {}

print('Performing DBSCAN for each driver...')
for driver, driverIndices in tqdm(driversIndicesMapped.items()):
    driverDistance = actualSetsDistances[driverIndices]
    driverDistance = csr_matrix(driverDistance[:, driverIndices])
    hdb = HDBSCAN(min_cluster_size=2, max_cluster_size=forward_expansion, metric="precomputed", store_centers=None, allow_single_cluster=False ).fit(driverDistance.copy())

    labels = hdb.labels_

    # Selecting the cluster that has the higher intra-cluster similarity
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
    # Access the data points in the selected cluster
    selected_cluster_points = driverDistance[labels == selected_cluster]
    selected_cluster_distance = intra_cluster_distance(selected_cluster_points)
    driver_best_divergence[driver] = np.min(selected_cluster_distance)
    driver_cluster_divergences[driver] = np.mean(selected_cluster_distance)
    # Find the medoid that maximize the inter-cluster similarity
    selected_point = np.argmin(selected_cluster_distance)
    best_clustroid = mapping_clustroid(labels, selected_cluster, selected_point)
    best_clustroids.append(best_clustroid)
    best_index = driversIndicesBack[driver][best_clustroid]
    selected_point = selected_cluster_points[selected_point]

    driversClustersPoints[driver] = selected_cluster_points
    driversLabels[driver] = labels
    driversClusterLabels[driver] = selected_cluster
    driversIdealPoint[driver] = selected_point
    driversIdealIndex[driver] = best_index

medoid_mean_distance = 0
medoid_best_distance = 0
divergence = 0
for driver in uniqueDrivers:
    medoid_mean_distance += driver_cluster_divergences[driver]
    medoid_best_distance += driver_best_divergence[driver]
    divergence += medoid_best_distance / medoid_mean_distance
medoid_mean_distance /= len(uniqueDrivers)
medoid_best_distance /= len(uniqueDrivers)
divergence /= len(uniqueDrivers)

print("TASK 3 RESULTS:")
print("\033[93m\nDriver's Divergence:\033[0m")
print(f"Mean Intra-Cluster Distance: {medoid_mean_distance}")
print(f"Best Intra-Cluster Distance: {medoid_best_distance}")
if divergence < 1:
    # print in green if the improvement is positive, in red if it is negative
    print("   Mean Divergence Improvement: \033[92m{:.2f}% \033[0m".format(-(1-divergence) * 100))
else:
    print("   Mean Divergence Decline: \033[91m{:.2f}% \033[0m\n".format((1-divergence) * 100))


data = []
for driver in uniqueDrivers:
    data.append({'driver':driver, 'route':dfActual.loc[driversIdealIndex[driver]]['route']})

# Write the data to the JSON file
print("\nWriting JSON data to results/perfectRoute" + SUFFIX_FILE + ".json...")
file_path = os.path.join(HOME, 'results', "perfectRoute" + SUFFIX_FILE + '.json') #"standard.json"
with open(file_path, 'w', encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=2, ensure_ascii=False)
    
print("JSON data has been written to results/perfectRoute" + SUFFIX_FILE + ".json")

print("\nTASK 3 FINISHED\n")

print("\n\nTotal time: {:.2f} seconds".format(time.time() - time_start))
    
if PLOT:
    
    import warnings

    # Disable warnings from the umap-learn library
    warnings.filterwarnings("ignore", category=UserWarning, module="umap")

    columns = 2
    rows = NUM_PLOTS // columns

    ## 2D Representation
    fig, axs = plt.subplots(rows, columns, figsize = (columns*3, rows*3))
    axs = axs.flatten()

    for i, driver in enumerate(uniqueDrivers):
        if i == NUM_PLOTS:
            break
        idx = best_clustroids[i]
        hdbscan_labels = driversLabels[driver]
        clustered = (hdbscan_labels >= 0)

        fit = umap.UMAP(n_neighbors=5,min_dist=0.9,n_components=2,metric='precomputed')
        driverDistance = actualSetsDistances[driversIndicesMapped[driver]]
        driverDistance = csr_matrix(driverDistance[:, driversIndicesMapped[driver]])
        u = fit.fit(driverDistance)

        axs[i].scatter(u.embedding_[clustered, 0], u.embedding_[clustered, 1], s = 10, c=hdbscan_labels[clustered], cmap='Spectral')

        # Noise
        axs[i].scatter(u.embedding_[~clustered, 0], u.embedding_[~clustered, 1], s = 10, color=(0.5,0.5,0.5), alpha=0.5)

        axs[i].plot(u.embedding_[idx, 0], u.embedding_[idx, 1], marker='+', markersize=10, color='black')

        axs[i].axhline(y=u.embedding_[idx, 1], color='gray', linestyle='--')
        axs[i].axvline(x=u.embedding_[idx, 0], color='gray', linestyle='--')

        axs[i].set_title(f"driver {driver}", fontsize=12)
        #axs[i].legend()

    plt.tight_layout()
    plt.show()


    ## 3D Representation
    fig = plt.figure(figsize=(columns * 3, rows * 3))

    for i, driver in enumerate(uniqueDrivers):
        if i == NUM_PLOTS:
            break
        idx = best_clustroids[i]
        hdbscan_labels = driversLabels[driver]
        clustered = (hdbscan_labels >= 0)

        fit = umap.UMAP(n_neighbors=5, min_dist=0.9, n_components=3, metric='precomputed')
        driverDistance = actualSetsDistances[driversIndicesMapped[driver]]
        driverDistance = csr_matrix(driverDistance[:, driversIndicesMapped[driver]])
        u = fit.fit(driverDistance)

        x = u.embedding_[clustered, 0]
        y = u.embedding_[clustered, 1]
        z = u.embedding_[clustered, 2]

        ax = fig.add_subplot(rows, columns, i + 1, projection='3d')
        ax.scatter(x, y, z, s=10, c=hdbscan_labels[clustered], cmap='Spectral')

        x_best = u.embedding_[idx, 0]
        y_best = u.embedding_[idx, 1]
        z_best = u.embedding_[idx, 2]

        ax.plot(x_best, y_best, z_best, marker='+', markersize=20, c='black')

        ax.set_title(f"driver {driver}", fontsize=12)

    plt.tight_layout()
    plt.show()
    
