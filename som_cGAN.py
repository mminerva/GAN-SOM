# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:28:35 2020

@author: sveta
"""

from minisom import MiniSom  

import matplotlib.pyplot as plt

import statistics

import os
import glob
from PIL import Image
import numpy as np

real_list = []
fake_list = []
#store name files 
legend_fake = []
legend_real = []

#ONLY THING YOU NEED TO CHANGE IS THE DATASET NAME TO RUN ON OTHER DATASET

#dataset = "edges2shoes"
#dataset = "facades_label2photo"
#dataset = "day2night"
#dataset = "map2sat"
dataset = "sat2map"
path = "cGAN_results/" + dataset + "_pretrained/test_latest/images/*.png"
#dataset = "horse2zebra/*.png"



def read_data():
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
   results = []
   size = len(data)
   
   for i in range(0,size):
      results.append(som.winner(data[i]))

   pairs = []
   for i in range(0, size):
      pairs.append([])
      for j in range(0, size):
         if results[i]==results[j]:
            pairs[i].append(j)

   return pairs

#figure out runtime error for quan error
#better code structure for more tests
#saving results
#same initial weights??
   
def fake_data():
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
   results = []
   size = len(data)
   
   for i in range(0,size):
      results.append(som.winner(data[i]))

   pairs = []
   for i in range(0, size):
      pairs.append([])
      for j in range(0, size):
         if results[i]==results[j]:
            pairs[i].append(j)

   return pairs

def combined_data():
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
   
   results = []
   
   for i in range(0,iteration):
      results.append(som.winner(data[i]))

   pairs = []
   for i in range(0, iteration):
      pairs.append([])
      for j in range(0, iteration):
         if results[i]==results[j]:
            pairs[i].append(j)

   return pairs

def compare(real, fake):
   res = []
   for i in range(0, len(real)):
      s1 = set(real[i])
      s2 = set(fake[i])
      res.append(len(s1.intersection(s2))/len(s1.union(s2)))
   
   return res

def convert_to_bitmask(int_array):
   '''Convert the integer array into a binary matrix with one row for each cluster, 
   indicating which element belongs to the specific cluster
   E.g. [0,1,1,2] -> [[1. 0. 0. 0.]
                      [0. 1. 1. 0.]
                      [0. 0. 0. 1.]]'''
   n_clusters = len(set(int_array))
   array_len = len(int_array)
   res = np.zeros((n_clusters, array_len))

   for c in range(0, n_clusters):
     mask = (np.array(int_array) == c).astype(int)
     res[c] = mask

   return res 

import itertools

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
   print(num_el_per_centroid)

   for i in range(0, len(same_el_per_centroid)):
     if num_el_per_centroid[i] > 1: # exclude groups with only one element (their score is 0 but it's not meaningful)
       tot += same_el_per_centroid[i]
       n_el += num_el_per_centroid[i]
   return (float(tot) / n_el)



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

iteration = 100

groups_real = real_data()
groups_fake = fake_data()
print(groups_real)
print(groups_fake)

comparison = compare(groups_real, groups_fake)
same_el_per_centroid, num_el_per_centroid = compare_2(groups_real, groups_fake)
comparison_2 = compare_aggr(same_el_per_centroid, num_el_per_centroid)

print("mean Jaccard similiarity: " + str(statistics.mean(comparison)))
print("median Jaccard similiarity: " + str(statistics.median(comparison)))
#print("mode Jaccard similiarity: " + str(statistics.mode(res)))
print("Number of elements with the same assignment in the fake domain, for each centroid: "+ str(same_el_per_centroid))
print("Number of elements in each real centroid: "+str(num_el_per_centroid))
print("Computed score: "+ str(comparison_2))

#f = open(dataset +"_results.txt", "w")
#f.write("mean Jaccard similiarity: " + str(statistics.mean(comparison)))
#f.write("median Jaccard similiarity: " + str(statistics.median(comparison)))
#f.write(str(sum) +  " pairs were grouped together out of " + str(int(size/2)))
#f.close()

res = combined_data()

size = len(res)

sum = 0
for i in range(0, int(size/2)):
   if int(i+50) in res[i]:
      sum = sum + 1
      
print(str(sum) +  " pairs were grouped together out of " + str(int(size/2)))
    

f = open(dataset +"_results.txt", "w")
print(dataset +"_results.txt")
f.write("mean Jaccard similiarity: " + str(statistics.mean(comparison)) + "\n")
f.write("median Jaccard similiarity: " + str(statistics.median(comparison)) + "\n")
f.write(str(sum) +  " pairs were grouped together out of " + str(int(size/2)) + "\n")
f.close()   
         



   