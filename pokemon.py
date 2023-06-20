import csv
import numpy as np
import scipy as sp

import math
import matplotlib.pyplot as plt

def load_data(filepath):
    with open(filepath, mode='r') as file:
        pokemon_dict = csv.DictReader(file)
        pokemon_dict = list(pokemon_dict)
    
    return pokemon_dict


def calc_features(row):
    x = np.array([row["Attack"], row["Sp. Atk"], row["Speed"], row["Defense"], row["Sp. Def"], row["HP"]], dtype="int64")
    return x


def hac(features):
    distance_matrix = create_dm(features)
    length = len(features)
    clusters = [[0 for i in range(4)] for j in range(length-1)]

    counter = 0
    for k in range(length):
        distance_matrix[k][k] = math.inf

    record_cluster = []
    #record_cluster = [id:#, clusters: pokemon]
    index_record = []
    id = 0
    boolean = False
    row_index = 0
    column_index = 0

    while(counter < len(clusters)):
        #find the minimum distance
        min = math.inf
        row_index = 0
        column_index = 0
        for i in range(length):
            for j in range(length):
                if(min > distance_matrix[i][j]):
                    boolean = False
                    for z in range(len(record_cluster)):
                        if i in record_cluster[z]["clusters"]:
                            if j in record_cluster[z]["clusters"]:
                                boolean = True

                    if(boolean):
                        continue
                    
                    temp1 = []
                    temp2 = []
                    max1 = -math.inf
                    if(i in index_record and j in index_record):
                        for z in range(len(record_cluster)):
                            if i in record_cluster[z]["clusters"]:
                                for w in range(len(record_cluster)):
                                    if j in record_cluster[w]["clusters"]:
                                        temp1 = record_cluster[z]['clusters']
                                        temp2 = record_cluster[w]['clusters']

                                        
                        for a in range(len(temp1)):
                            for b in range(len(temp2)):
                                if(distance_matrix[temp1[a]][temp2[b]] > max1):
                                        max1 = distance_matrix[temp1[a]][temp2[b]]                

                        if(max1 < min):
                            row_index = i
                            column_index = j
                            min = max1
                            break
                        else:
                            break
                    

                    elif(i in index_record or j in index_record):
                        max = get_max(i, j, record_cluster, distance_matrix)
                        if(max > min):
                            pass
                        else:
                            row_index = i
                            column_index = j
                            min = max
                    else:
                        min = distance_matrix[i][j]
                        row_index = i
                        column_index = j


        
        distance = min

        index_record.append(row_index)
        index_record.append(column_index)
        toLeave = False
        if(check_clus(record_cluster, row_index) and check_clus(record_cluster, column_index)):
            id1 = 0
            id2 = 0
            l1 = []
            for z in range(len(record_cluster)):
                if(toLeave):
                    break
                if column_index in record_cluster[z]["clusters"]:
                    id1 = record_cluster[z]['id']
                    for g in range(len(record_cluster)):
                        if(toLeave):
                            break
                        if row_index in record_cluster[g]["clusters"]:
                            id2 = record_cluster[g]['id']
                            l1 = record_cluster[g]['clusters'] + record_cluster[z]['clusters']
                            if(g>z):
                                del record_cluster[g]
                                del record_cluster[z]
                            else:
                                del record_cluster[z]
                                del record_cluster[g]
                            toLeave = True
                            break

            record_cluster.append({'id':counter, 'clusters': l1})
            sumpoke = clusters[id1][3] + clusters[id2][3]
            if(length + id1 <= length + id2):
                clusters[counter] = [length + id1, length+ id2, distance, sumpoke]
            else:
                clusters[counter] = [length + id2, length+ id1, distance, sumpoke]

        elif(not check_clus(record_cluster, row_index) and check_clus(record_cluster, column_index)):
            id1 = 0
            previd = 0
            for z in range(len(record_cluster)):
                if column_index in record_cluster[z]["clusters"]:
                    id1 = record_cluster[z]['id']

                    
                    record_cluster[z]["clusters"].append(row_index)
                    previd = record_cluster[z]["id"]
                    record_cluster[z]["id"] = counter
                    break
            
            clusters[counter] = [row_index, length + previd, distance, clusters[id1][3] + 1]


        elif(check_clus(record_cluster, row_index) and not check_clus(record_cluster, column_index)):
            previd = 0
            id1 = 0
            for z in range(len(record_cluster)):
                if row_index in record_cluster[z]["clusters"]:
                    id1 = record_cluster[z]['id']



                record_cluster[z]["clusters"].append(column_index)
                previd = record_cluster[z]["id"]
                record_cluster[z]["id"] = counter
                break


            clusters[counter] = [column_index, length + previd, distance, clusters[id1][3] + 1]

        
        else:
            if(row_index <= column_index):
                clusters[counter] = [row_index, column_index, distance, 2]
            else:
                clusters[counter] = [column_index, row_index, distance, 2]
            record_cluster.append({'id': counter, 'clusters' : [row_index, column_index]})
            


        counter += 1
    return np.array(clusters)

    

def get_max(i, j, rc, dm):
    second = None
    first = None
    for q in range(len(rc)):
        if i in rc[q]['clusters']:
            first = rc[q]['clusters']
        if j in rc[q]['clusters']:
            second = rc[q]['clusters']
    
    max = -math.inf

    if (first == None):
        for y in range(len(second)):
            if (max < dm[i][second[y]]):
                max = dm[i][second[y]]
        return max

    if (second == None):
            for y in range(len(first)):
                if (max < dm[first[y]][j]):
                    max = dm[first[y]][j]
            return max

    
    for x in range(len(first)):
        for y in range(len(second)):
            if (max < dm[first[x]][second[y]]):
                max = dm[first[x]][second[y]]
    return max


def check_min(dm, rc, i, j, min):
    for z in range(len(rc)):
        if i in rc[z]["clusters"]:
            for x in range(len(rc[z]["clusters"])):
                if rc[z]["clusters"][x] >= min:
                    return False
        if j in rc[z]["clusters"]:
            for x in range(len(rc[z]["clusters"])):
                if rc[z]["clusters"][x] >= min:
                    return False


    return True

def check_clus(rc, i):
    for z in range(len(rc)):
        if i in rc[z]["clusters"]:
            return True
        
    return False



def create_dm(features):
    length = len(features)
    distance_matrix = [[0 for i in range(length)] for j in range(length)]

    transposed_features = np.transpose(features)
    germian_matrix = np.dot(features, transposed_features)


    features_squared = []
    for i in range(length):
        temp = features[i]**2
        features_squared.append([sum(temp)])
    
    transposed_features_squared = np.transpose(features_squared)
    one_vector = [[1]for j in range(length)]
    transposed_one_vector = np.transpose(one_vector)

    comp1 = np.matmul(features_squared, transposed_one_vector)
    comp2 = np.matmul(one_vector, transposed_features_squared)
    
    distance_matrix = comp1 + comp2 - 2*germian_matrix
    distance_matrix = np.sqrt(distance_matrix)
    return distance_matrix


def imshow_hac(Z, names):
    fig, ax1 = plt.subplots(1)
    ax1 = sp.cluster.hierarchy.dendrogram(Z, labels = names, leaf_rotation = "vertical")
    plt.tight_layout()
    c = len(names)
    fig.set_size_inches(12.5, 11.5)
    plt.title("N = " + str(c), loc="center")
    plt.show() 


print(load_data('Pokemon.csv'))