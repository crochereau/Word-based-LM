from paths import WIKIPEDIA_HOME
from paths import CHAR_VOCAB_HOME
from paths import FIGURES_HOME
from paths import MODELS_HOME

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--char_embedding_size", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--layer_num", type=int, default=1)
parser.add_argument("--weight_dropout_in", type=float, default=0.01)
parser.add_argument("--weight_dropout_hidden", type=float, default=0.1)
parser.add_argument("--char_dropout_prob", type=float, default=0.33)
parser.add_argument("--char_noise_prob", type = float, default= 0.01)
parser.add_argument("--learning_rate", type = float, default= 0.1)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=50)


args=parser.parse_args()
print(args)





import corpusIteratorWiki



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

try:
   with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
    print("Creating new vocab")
    char_counts = {}
    # get symbol vocabulary

    with open(WIKIPEDIA_HOME+"/"+args.language+"-vocab.txt", "r") as inFile:
      words = inFile.read().strip().split("\n")
      for word in words:
         for char in word.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
    char_counts = [(x,y) for x, y in char_counts.items()]
    itos = [x for x,y in sorted(char_counts, key=lambda z:(z[0],-z[1])) if y > 50]
    with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "w") as outFile:
       print("\n".join(itos), file=outFile)
#itos = sorted(itos)
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()


named_modules = {"output" : output}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])
else:
   assert False


embeddings = output.weight.data.cpu().numpy()

import torch
import tsne
import json
import sys
import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')

X = embeddings 
print(X)
Y = tsne.tsne(X=X, no_dims=2, initial_dims=100, perplexity=30.0)
fig, ax = plt.subplots()
for i in range(len(itos)):
   ax.annotate(itos[i], (Y[i+3,0], Y[i+3,1]))
ax.scatter(Y[:, 0], Y[:, 1], 20) #, [5.0 for _ in range(len(itos))])

plt.show()
plt.savefig(f"chars-t-sne-{args.language}.png") 
        #pylab.scatter(Y[:, 0], Y[:, 1], 20, [5.0 for _ in range(50)])
        #pylab.show()



quit()



import scipy.cluster.hierarchy
import sklearn.cluster

clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=10, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward')
y = clustering.fit_predict(embeddings)
from scipy.cluster.hierarchy import dendrogram
print(y)
for i, char in enumerate(itos):
   print("\t".join([char, str(y[i+3])]))


import numpy as np



import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

labels = ["EOS", "_", "OOV"] + itos

children = clustering.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
dendrogram(linkage_matrix, labels=labels)


plt.show()
plt.savefig(FIGURES_HOME+"/figures/char-emb-clustering-output_output-"+args.load_from+".png")
plt.close()



