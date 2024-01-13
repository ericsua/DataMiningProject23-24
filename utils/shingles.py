import numpy as np

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

def hash_shingles(shingles):
    # hash shingles
    string = ""
    for shingle in shingles:
        string += str(shingle) + ","
    return hash(string)

def create_shingles_selfcontained(s, k, uniqueCities, uniqueItems, longestRoute, maxItemQuantity):
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