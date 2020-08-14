#==================================================
#Implementation of the k-means clustering algorithm
#SE Level 3, Task 22
#Vivek Harase
#==================================================

import os  
from math import sqrt  
import random  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  
pd.options.display.max_rows = 4000  


def read_csv_pd(filename):
    #This function reads a csv file with pandas, prints the dataframe and returns
    #the two columns in numpy ndarray for processing as well as the country names in
    #numpy array needed for cluster matched results
    data1 = pd.read_csv((filename), delimiter=',')
    print(data1)
    country_names = data1[data1.columns[0]].values
    list_array = data1[[
        data1.columns[1], data1.columns[2]]].values
    return list_array, country_names


def distance_between(cent, data_points):
    #This function calculates the euclidean distance between each data point and each centroid.
    #It appends all the values to a list and returns this list
    distances_arr = []  
    for centroid in cent:
        for datapoint in data_points:
            distances_arr.append(
                sqrt((datapoint[0]-centroid[0])**2 + (datapoint[1]-centroid[1])**2))
    return distances_arr

x = read_csv_pd("data1953.csv")
# convert the ndarray to a list for sampling
x_list = np.ndarray.tolist(x[0][0:, :])
# User input
k = int(input("Please enter the number of clusters you want: "))
iterations = int(
    input("Please enter the number of iterations that the algorithm must run: "))

centroids = random.sample(x_list, k)
print('Random Centroids are: ' + str(centroids))


def assign_to_cluster_mean_centroid(x_in=x, centroids_in=centroids, n_user=k):
#This function calls the distance_between() function. It allocates from
#the returned list, each data point to the centroid/cluster that it is the
#closest to in distance. It also rewrites the centroids with the newly calculated
#means. It then returns the list with cluster allocations that are 
#in line with the order of the countries. It also returns the dictionary with clusters
    distances_arr_re = np.reshape(distance_between(
        centroids_in, x_in[0]), (len(centroids_in), len(x_in[0])))
    datapoint_cen = []
    distances_min = []
    for value in zip(*distances_arr_re):
        distances_min.append(min(value))
        datapoint_cen.append(np.argmin(value)+1)
    #Creation of Cluster dictionary to add number of clusters according to user input    
    clusters = {}
    for no_user in range(0, n_user):
        clusters[no_user+1] = []

    #Allocation of each datapoint to its closest cluster
    for d_point, cent in zip(x_in[0], datapoint_cen):
        clusters[cent].append(d_point)

    #This for loop is run to rewrite the centroids with the new mean
    for i, cluster in enumerate(clusters):
        reshaped = np.reshape(clusters[cluster], (len(clusters[cluster]), 2))
        centroids[i][0] = sum(reshaped[0:, 0])/len(reshaped[0:, 0])
        centroids[i][1] = sum(reshaped[0:, 1])/len(reshaped[0:, 1])
    print('Centroids for this iteration are:' + str(centroids))
    return datapoint_cen, clusters


# Scatterplot of the data without clustering
plt.scatter(x[0][0:, 0], x[0][0:, 1])
plt.xlabel('Birthrate')
plt.ylabel('Life Expectancy')
plt.title('Data Points with random centroids\nNo data point allocation')
cv = np.reshape(centroids, (k, 2))
plt.plot(cv[0:, 0], cv[0:, 1],
         c='#000000', marker="*", markersize=15, linestyle=None, linewidth=0)
plt.show()

# sets the font size of the labels on matplotlib
plt.rc('font', size=14)
# sets style of plots
sns.set_style('white')
# define a custom pallette
customPalette = ['#FF8000', '#660198', '#006400', '#FF0000',
                 '#FF0000', '#61B329', '#E04006', '#FBDB0C',
                 '#FF6600', '#FF8000', '#8B4789', '#EE7AE9']
sns.set_palette(customPalette)

# ========================
# MAIN LOOP - Comp Task 2
# ========================

for iteration in range(0, iterations):
    
    print("ITERATION: " + str(iteration+1))
    assigning = assign_to_cluster_mean_centroid() #Assigns the function to a variable because it has more than one return value

    #Creates a dataframe for visualisation
    cluster_data = pd.DataFrame({'Birth Rate': x[0][0:, 0],'Life Expectancy': x[0][0:, 1],'label': assigning[0],'Country': x[1]})

    #Creation of dataframe and grouping to print inferences
    group_by_cluster = cluster_data[[
        'Country', 'Birth Rate', 'Life Expectancy', 'label']].groupby('label')
    count_clusters = group_by_cluster.count()

    # Inference 1
    print("COUNTRIES PER CLUSTER: \n" + str(count_clusters))

    # Inference 2
    print("LIST OF COUNTRIES PER CLUSTER: \n",
          list(group_by_cluster))

    # Inference 3
    print("AVERAGES: \n", str(cluster_data.groupby(['label']).mean()))
    mean = assigning[1]
    means = {}
    for clst in range(0, k):
        means[clst+1] = []

    # Creating a for loop to calculate the squared distances between each
    # data point and its cluster mean
    for index, data in enumerate(mean):
        array = np.array(mean[data])
        array = np.reshape(array, (len(array), 2))
        
        birth_rate = sum(array[0:, 0])/len(array[0:, 0])
        life_exp = sum(array[0:, 1])/len(array[0:, 1])
        
        for data_point in array:
            distance = sqrt(
                (birth_rate-data_point[0])**2+(life_exp-data_point[1])**2)
            means[index+1].append(distance)
    # creating a list that will hold all the sums of the means in each of the clusters.
    total_distance = []
    for ind, summed in enumerate(means):
        total_distance.append(sum(means[ind+1]))

    print("Summed distance of all clusters: " + str(sum(total_distance)))
    facet = sns.lmplot(data=cluster_data, x='Birth Rate', y='Life Expectancy', hue='label',
                       fit_reg=False, legend=False, legend_out=False)
    plt.legend(loc='upper right')
    centr = np.reshape(centroids, (k, 2))
    plt.plot(centr[0:, 0], centr[0:, 1], c='#000000', marker="*", markersize=15, linestyle=None, linewidth=0)
    plt.title('Iteration: ' + str(iteration+1) + "\nSummed distance of all clusters: " + str(round(sum(total_distance), 0)))
    plt.show()
