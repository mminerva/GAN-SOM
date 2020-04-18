# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:28:35 2020

@author: sveta
"""

from minisom import MiniSom  

import itertools

import matplotlib.pyplot as plt

import statistics

import os
import glob
from PIL import Image
import numpy as np

#holders for datasets
real_list = []
fake_list = []
#store name files 
legend_fake = []
legend_real = []

#ONLY THING YOU NEED TO CHANGE IS THE DATASET NAME TO RUN ON OTHER DATASET

dataset = "edges2shoes"
#dataset = "facades_label2photo"
#dataset = "day2night"
#dataset = "map2sat"
#dataset = "sat2map"
path = "cGAN_results/" + dataset + "_pretrained/test_latest/images/*.png"
#dataset = "horse2zebra/*.png"



def read_data():
   #reads in data from the defined path, stores image data in real_list and fake_list depending on image name
   #stores the image name in corresponding legend
   #images are read in order, so the fake-real pairs are in same order in their corresponding lists
   print("results for dataset " + dataset)
   for filename in glob.glob(path): 
       im=Image.open(filename)
       im.load()
       data = np.asarray(im, dtype= "int32")
       filename = filename[51:-4]
       if "real_B" in filename:
          real_list.append(data.flatten())
          legend_real.append(filename)
       if "fake_B" in filename:
         fake_list.append(data.flatten())
         legend_fake.append(filename)
         
            
def real_data(): 
   #performs SOM on the real data only
   #shows and saves an image of the frequency distribution of nodes being winners
   #results variable stores the winner coordinates for each image
   #the winner coordinates are essentially the label of the cluster the image belongs to
   #returns array where for each image there is a list of other images it was clustered together with in this SOM
   
   data = real
   
   #parameter for som training is number of iterations, should be comparable to number of samples for best results
   iteration = len(data)
   #train som
   som = MiniSom(5, 5, pixels, sigma=0.5, learning_rate=0.5)
   som.random_weights_init(data)
   som.train_random(data, iteration)
   
   #plt.pcolor(som.distance_map().T, cmap='bone_r')
   
   #show some results
   
   print("results for real data\n")
   
   fig = plt.figure(figsize=(8, 7))
   frequencies = som.activation_response(data)
   plt.pcolor(frequencies.T, cmap='Blues') 
   plt.colorbar()
   
    
   name = dataset + "_som_map_real.png"
   plt.savefig(name)
   
   plt.show()

   
   plt.close(fig)
   
   #act = som.activation_response(data)
   #print(act)
   
   #print(som.quantization_error(data))
   
   #print(som.topographic_error(data))
   
   #print(som.winner(data[1]))
   #som.labels_map(data, legend_real)
   
   #collect winner coordinates for each image
   results = []
   size = len(data)
   
   for i in range(0,size):
      results.append(som.winner(data[i]))
      
   #count number of clusters
   clusters = []
   for i in range(0, iteration):
      clusters.append([])
      for j in range(0, iteration):
         if results[i]==results[j]:
            clusters[i].append(j)
            
   n = len(np.unique(clusters))


   #collect the image indexes for which images each image was grouped for, except for itself
   pairs = []
   for i in range(0, size):
      pairs.append([])
      for j in range(0, size):
         if not(i==j):
            if results[i]==results[j]:
               pairs[i].append(j)

   return pairs, n, np.unique(clusters)

#figure out runtime error for quan error
#same initial weights??
   
def fake_data():
   #performs SOM on the fake data only
   #shows and saves an image of the frequency distribution of nodes being winners
   #results variable stores the winner coordinates for each image
   #the winner coordinates are essentially the label of the cluster the image belongs to
   #returns array where for each image there is a list of other images it was clustered together with in this SOM
   
   data = fake
   
   #parameter for som training is number of iterations, should be comparable to number of samples for best results
   iteration = len(data)
   
   
   #train som
   som = MiniSom(5, 5, pixels, sigma=0.5, learning_rate=0.5)
   som.random_weights_init(data)
   som.train_random(data, iteration)
   
   #plt.pcolor(som.distance_map().T, cmap='bone_r')
   
   #show some results
   
   print("results for fake data\n")
   
   fig = plt.figure(figsize=(8, 7))
   frequencies = som.activation_response(data)
   plt.pcolor(frequencies.T, cmap='Blues') 
   plt.colorbar()
   
   name = dataset + "_som_map_fake.png"
   plt.savefig(name)
   
   
   plt.show()
   
   
   plt.close(fig)
   
   #act = som.activation_response(data)
   #print(act)
   
   #print(som.quantization_error(data))
   
   #print(som.topographic_error(data))
   
   #print(som.winner(data[1]))
   #som.labels_map(data, legend_fake)
   
   #collect winner coordinates for each image
   results = []
   size = len(data)
   
   for i in range(0,size):
      results.append(som.winner(data[i]))
      
      
      #count number of clusters
   clusters = []
   for i in range(0, iteration):
      clusters.append([])
      for j in range(0, iteration):
         if results[i]==results[j]:
            clusters[i].append(j)
            
   n = len(np.unique(clusters))


   #collect the image indexes for which images each image was grouped for, except for itself
   pairs = []
   for i in range(0, size):
      pairs.append([])
      for j in range(0, size):
         if not(i==j):
            if results[i]==results[j]:
               pairs[i].append(j)

   return pairs, n, np.unique(clusters)

def combined_data():
   #performs SOM on the combined data
   #shows and saves an image of the frequency distribution of nodes being winners
   #results variable stores the winner coordinates for each image
   #the winner coordinates are essentially the label of the cluster the image belongs to
   #returns array where for each image there is a list of other images it was clustered together with in this SOM
   data = all_list
   
   #parameter for som training is number of iterations, should be comparable to number of samples for best results
   iteration = len(data)
   #train som
   som = MiniSom(5, 5, pixels, sigma=0.5, learning_rate=0.5)
   som.random_weights_init(data)
   som.train_random(data, iteration)
   
   #plt.pcolor(som.distance_map().T, cmap='bone_r')
   
   #show some results
   
   print("results for combined data\n")
   
   fig = plt.figure(figsize=(8, 7))
   frequencies = som.activation_response(data)
   plt.pcolor(frequencies.T, cmap='Blues') 
   plt.colorbar()
   
   name = dataset + "_som_map_combined.png"
   plt.savefig(name)
   
   
   plt.show()

   plt.close(fig)
   
   #act = som.activation_response(data)
   #print(act)
   
   #print(som.quantization_error(data))
   
   #print(som.topographic_error(data))
   
   #print(som.winner(data[1]))
   
   #collect winner coordinates for each image
   results = []
   
   for i in range(0,iteration):
      results.append(som.winner(data[i]))
      
   #count number of clusters
   clusters = []
   for i in range(0, iteration):
      clusters.append([])
      for j in range(0, iteration):
         if results[i]==results[j]:
            clusters[i].append(j)
            
   n = len(np.unique(clusters))



   #collect the image indexes for which images each image was grouped for, except for itself
   pairs = []
   for i in range(0, iteration):
      pairs.append([])
      for j in range(0, iteration):
         if not(i==j):
            if results[i]==results[j]:
               pairs[i].append(j)

   return pairs, n, np.unique(clusters)

def compare(real, fake):
   #takes in two data sets of same size
   #returns vector of element-wize Jaccard set similiarity
   res = []
   for i in range(0, len(real)):
      s1 = set(real[i])
      s2 = set(fake[i])
      if not(len(s1.union(s2))==0):
         res.append(len(s1.intersection(s2))/len(s1.union(s2)))
            
   
   return res

def compare_2(real_d, fake_d):
   '''Compare assignments by couples.
   Returns:
   - same_el_per_centroid = number of couples that have the same assignment in the real and fake domain;
                        the result is a vector: couples are grouped by real centroid. In other words, 
                        each cell of this array is the number of couples that have real image in that group, and have the same assignment for the fake domain
   - num_el_per_centroid = number of images grouped in this centroid'''
   #this eliminates duplicates
   real_d.sort()
   real = list(real_d for real_d,_ in itertools.groupby(real_d))
   fake_d.sort()
   fake = list(fake_d for fake_d,_ in itertools.groupby(fake_d))
   
   same_el_per_centroid = np.zeros(len(real))
   num_el_per_centroid = np.zeros(len(real))

   for i in range(0, len(real)):
     num_el_per_centroid[i] = len(real[i])
     for j in range(0, len(fake)):
       s1 = set(real[i])
       s2 = set(fake[j]) 
       score =  len(s1.intersection(s2)) if  (len(s1.intersection(s2)) > 1) else 0 # increment the score only if there is more than one match between the two sets
       same_el_per_centroid[i] += score 

   return same_el_per_centroid, num_el_per_centroid

def compare_aggr(same_el_per_centroid, num_el_per_centroid):
   '''Aggregate the result from  compare_2().
   Returns the number of couples that are grouped together over the total number. [value between 0 and 1]
   Takes out from the computation the couples that are clustered alone, since their score is 0 but they cannot have more.'''
   tot = 0
   n_el = 0
   #print(num_el_per_centroid)

   for i in range(0, len(same_el_per_centroid)):
     if num_el_per_centroid[i] > 1: # exclude groups with only one element (their score is 0 but it's not meaningful)
       tot += same_el_per_centroid[i]
       n_el += num_el_per_centroid[i]
   return (float(tot) / n_el)

def compare_clusters(real, fake):
   #takes in two vectors of clusters
   #could be different size
   #only contain unique elements
   #each element is itself a list of image indexes in that cluster
   #can consider image 1 in real dataset the same as image 1 in fake dataset as they are the corresponding pair
   
   #need to compare all clusters with all other clusters 
   
   res = []
   for i in range(0, len(real)):
      res.append([])
      for j in range(0, len(fake)):
         if not(i==j):
            s1 = set(real[i])
            s2 = set(fake[j])
            if not(len(s1.union(s2))==0):
                  res[i].append(len(s1.intersection(s2))/len(s1.union(s2)))
            
   return res

   
read_data() 
   
#domain B
fake = np.array(fake_list)
#print(legend_fake)
#print(len(legend_fake))
#print(fake.shape)
   
#domain A
real = np.array(real_list)
#print(legend_real)
#print(len(legend_real))
#print(real.shape)
   
#combined dataset
images = []
images.extend(fake_list)
images.extend(real_list)
all_list = np.array(images)
   
legend = []
legend.extend(legend_fake)
legend.extend(legend_real)
#print(legend)
#print(len(legend))
#print(all_list.shape)

#number of parameters in a sample
pixels = len(fake[1])
#print(pixels)

#iteration = 100

#fit real and fake datasets separately
groups_real, nr, clusters_real  = real_data()
groups_fake, nf, clusters_fake  = fake_data()

#perform metrics on the datasets
#this measures for each image how are its images it was paired with compares between the two mappings
comparison = compare(groups_real, groups_fake)

#this meausures how similar are the clusters between the two mappings
cluster_comparison = compare_clusters(clusters_real, clusters_fake)


#print results of the metrics
#print(groups_real)
print("real SOM has "+ str(nr) + " clusters\n")
#print(groups_fake)
print("fake SOM has "+ str(nf) + " clusters\n")
#print(clusters_real)
#print(clusters_fake)
#print(comparison)
#print(cluster_comparison)
print("mean Jaccard similiarity for images: " + str(statistics.mean(comparison)))
print("median Jaccard similiarity for images: " + str(statistics.median(comparison)))
#print("mode Jaccard similiarity: " + str(statistics.mode(res)))

#convolute the cluster similiarities
cluster_averages = []
for i in range(0, len(cluster_comparison)):
   cluster_averages.append(max(cluster_comparison[i]))
   
#print(cluster_averages)

print("mean Jaccard similiarity of clusters: " + str(statistics.mean(cluster_averages)))
print("median Jaccard similiarity of clusters: " + str(statistics.median(cluster_averages)))
#print("mode Jaccard similiarity: " + str(statistics.mode(res)))


#do metrics on combined dataset
res, n, clusters = combined_data()

#print(res)
print("combined SOM has "+ str(n) + " clusters\n")
#print(clusters)

size = len(res)

sum = 0
for i in range(0, int(size/2)):
   if int(i+50) in res[i]:
      sum = sum + 1
      
print(str(sum) +  " pairs were grouped together out of " + str(int(size/2)))

#calculate how many clusters are solely from real or from fake dataset
fake_clusters = []
real_clusters = []
for i in range(0, len(clusters)):
   if all(j >= int(size/2) for j in clusters[i]):
      real_clusters.append(clusters[i])
   if all(j < int(size/2) for j in clusters[i]):
      fake_clusters.append(clusters[i])

#print(real_clusters)
#print(fake_clusters)
      
print("There are " + str(len(real_clusters)) + " real image only clusters")
print("There are " + str(len(fake_clusters)) + " fake image only clusters")


#additional metric
#print(clusters_real)
#print(clusters_fake)
same_el_per_centroid, num_el_per_centroid = compare_2(clusters_real, clusters_fake)
comparison_2 = compare_aggr(same_el_per_centroid, num_el_per_centroid)

print("\nNumber of elements with the same assignment in the fake domain, for each centroid: "+ str(same_el_per_centroid))
print("Number of elements in each real centroid: "+str(num_el_per_centroid))
print("Computed score: "+ str(comparison_2))
    


#save results
f = open(dataset +"_results.txt", "w")
f.write("real SOM has "+ str(nr) + " clusters\n")
f.write("real SOM has "+ str(nf) + " clusters\n")
f.write("combined SOM has "+ str(n) + " clusters\n")
f.write("mean Jaccard similiarity: " + str(statistics.mean(comparison)) + "\n")
f.write("median Jaccard similiarity: " + str(statistics.median(comparison)) + "\n")
f.write("mean Jaccard similiarity of clusters: " + str(statistics.mean(cluster_averages)))
f.write("median Jaccard similiarity of clusters: " + str(statistics.median(cluster_averages)))
f.write(str(sum) +  " pairs were grouped together out of " + str(int(size/2)) + "\n")
f.write("There are " + str(len(real_clusters)) + " real image only clusters")
f.write("There are " + str(len(fake_clusters)) + " fake image only clusters")
f.close()   
         



   