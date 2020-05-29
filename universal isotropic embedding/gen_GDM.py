#!/usr/bin/env python

import collections
from operator import itemgetter
import re
import nltk
import numpy as np
from nltk.corpus import stopwords                                  
from itertools import permutations
import matplotlib.pyplot as plt
import word2vec
import pandas as pd
from numpy import save
import copy
import networkx as nx


text_content = open('test1000.txt', encoding = 'utf-8', errors = 'ignore').read()


def tokenize(text):
    """
    Will split a text into words and remove stop words
    :param text: string - can be a document or a query
    :return: list - a list of words
    """
    words = None
    # the text into a sequence of words and store it in words
    words = text.split()
    
    # # remove stopwords
    # words = [word for word in words if word not in stopwords.words('english')]
    
    
    return words


def index(cleaned_texts):
    """
    use a dictionary (HashMap) for storing the inverted index
    key: word
    value: index of the word (vertx number) in the vertex list 
    """
    split_texts = tokenize(cleaned_texts) 
    inverted_index = {}
    split_text_set = set(split_texts)
    vertex_list = []
    for index, word in enumerate(split_text_set):
            inverted_index[word] = inverted_index.get(word, [])
            inverted_index[word].append(index)
            vertex_list.append(word)
    return inverted_index, vertex_list, split_texts


def generate_edge(word_list, n):
    """
    use a sliding window of size n, count the number of pairs as edge weights
    input size: n
    return: nxn matrix
    """
    edge_mat = np.zeros([max(word_list)+1, max(word_list)+1])
    pair_list = []

    for i in range(len(word_list)-n+1):
        pair_list.append(list(permutations(word_list[i:i+n],2))) 

    for i in range(len(pair_list)):
        for it in pair_list[i]:
            edge_mat[it[0],it[1]] += 1
    return edge_mat # (a,b), (b,a) counted twice
    



# preprocess 

# range for sliding window length 
N_RANGE = [3,5]#range(2,12,2)

cleaned_texts = []

regex = re.compile('[^a-zA-Z]') # filter only letters
for innerText in text_content:
    cleaned_texts.append(regex.sub(' ', innerText))  

# convert into lowercase
for index in range(0, len(cleaned_texts)):
    cleaned_texts[index] = cleaned_texts[index].lower()
cleaned_texts = ''.join(cleaned_texts) # convert back to str


inverted_index, vertex_list,split_texts = index(cleaned_texts)
decoded_texts = [inverted_index[word][0] for word in split_texts]

# generate edge weight matrix 
edge_mat_dict = {}
print('decoded_texts length:',len(decoded_texts))


for n in N_RANGE:
    edge_mat_dict[n] = generate_edge(decoded_texts, n)
    print(n,sum(sum(edge_mat_dict[n])))
    np.save('edge_mat'+str(n)+'.npy', edge_mat_dict[n])


GDM_dict0 = copy.deepcopy(edge_mat_dict)

# assign not connected with inf
for n in N_RANGE:
    GDM_dict0[n][GDM_dict0[n] == 0] = 1e30
    # dignal = 0
    for i in range(len(GDM_dict0[n])):
        GDM_dict0[n][i,i] = 0

GDM_dict = {} # key: n, value: GDM

eps = 1e-5
for n in N_RANGE:
    print('*********WINDOW LENGTH = ', str(n), '**********')
    GDM = GDM_dict0[n]
    
    
    G0 = nx.Graph()
    G0.add_nodes_from(range(len(vertex_list)))
    edge_mat = edge_mat_dict[n]
    # flatten the matrix as [x_idx,y_idx, value]
    XX,YY = np.meshgrid(np.arange(edge_mat.shape[1]),np.arange(edge_mat.shape[0]))
    idx = [edge_mat.ravel() != 0]
    flattened_mat = np.vstack((XX.ravel()[idx], YY.ravel()[idx], edge_mat.ravel()[idx])).T
    flattened_mat = [tuple(ele) for ele in flattened_mat]
    G0.add_weighted_edges_from(flattened_mat)
    
    for source in range(len(inverted_index.keys())):
        if source % 100 == 0:
            print(source/len(inverted_index.keys()))
        for target in range(len(inverted_index.keys())):
            if source != target:
                GDM[source,target] = nx.dijkstra_path_length(G0,source,target)

    GDM_dict[n] = GDM
    np.save('GDM'+str(n)+'.npy', GDM)

#    ## verify GDM symmetry
#    for u in range(len(GDM)):
#        for v in range(len(GDM)):
#            if abs(GDM[u,v] - GDM[v,u]) > eps:
#                print("GDM[%d,%d] != GDM[%d,%d]\n", u,v, v,u)
#
#    print('test symmetry done')
#
#    ## test triangle inequalities
#    for w in range(len(GDM)):
#        for u in range(len(GDM)):
#            for v in range(len(GDM)):
#                if GDM[v,u] + GDM[u,w] < GDM[v,w] - eps:
#                    print("GDM[%d,%d] + GDM[%d,%d] < GDM[%d,%d]\n", v,u,u,w,v,w)
#    print('test triangle inequalities done')



def word2GDM(text_content, N_RANGE):
    
    # preprocessing
    cleaned_texts = []
    regex = re.compile('[^a-zA-Z]') # filter only letters
    for innerText in text_content:
        cleaned_texts.append(regex.sub(' ', innerText))

    # convert into lowercase
    for index in range(0, len(cleaned_texts)):
        cleaned_texts[index] = cleaned_texts[index].lower()
    cleaned_texts = ''.join(cleaned_texts) # convert back to str

    inverted_index, vertex_list,split_texts = index(cleaned_texts)
    decoded_texts = [inverted_index[word][0] for word in split_texts]

    # generate edge weight matrix
    edge_mat_dict = {}
#    print('decoded_texts length:',len(decoded_texts))

    for n in N_RANGE:
        edge_mat_dict[n] = generate_edge(decoded_texts, n)
#        print(n,sum(sum(edge_mat_dict[n])))
#        np.save('edge_mat'+str(n)+'.npy', edge_mat_dict[n])

    GDM_dict0 = copy.deepcopy(edge_mat_dict)

    # assign not connected with inf
    for n in N_RANGE:
        GDM_dict0[n][GDM_dict0[n] == 0] = 1e30
        # dignal = 0
        for i in range(len(GDM_dict0[n])):
            GDM_dict0[n][i,i] = 0

    GDM_dict = {} # key: n, value: GDM

    eps = 1e-5
    for n in N_RANGE:
        print('*********WINDOW LENGTH = ', str(n), '**********')
        GDM = GDM_dict0[n]
        # update non-connected vertices with shortest path
        for z in range(len(GDM)):
            for u in range(len(GDM)):
                for v in range(len(GDM)):
                    if GDM[u,v] > GDM[u,z] + GDM[z,v]:
                        GDM[u,v] = GDM[u,z] + GDM[z,v]

    GDM_dict[n] = GDM
    np.save('GDM'+str(n)+'.npy', GDM)

    # ## verify GDM symmetry
    # for u in range(len(GDM)):
    #     for v in range(len(GDM)):
    #         if abs(GDM[u,v] - GDM[v,u]) > eps:
    #             print("GDM[%d,%d] != GDM[%d,%d]\n", u,v, v,u)

    # print('test symmetry done')

    # ## test triangle inequalities
    # for w in range(len(GDM)):
    #     for u in range(len(GDM)):
    #         for v in range(len(GDM)):
    #             if GDM[v,u] + GDM[u,w] < GDM[v,w] - eps:
    #                 print("GDM[%d,%d] + GDM[%d,%d] < GDM[%d,%d]\n", v,u,u,w,v,w)
    #     print('test triangle inequalities done')

    return GDM_dict


