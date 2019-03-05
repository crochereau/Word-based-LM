from paths import WIKIPEDIA_HOME
from paths import CHAR_VOCAB_HOME
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
import random

import math

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
print("".join(itos))
#quit()
stoi = dict([(itos[i],i) for i in range(len(itos))])

itosSet = set(itos)


import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


from torch.autograd import Variable


import time
training_data = corpusIteratorWiki.training(args.language)
bigrams = {}
unigrams = {}
counter = 0
for chunk in training_data:
   for i in range(0, len(chunk)-1):
#     if chunk[i] < "a" or chunk[i] > "ü":
#         continue
#     if chunk[i+1] < "a" or chunk[i+1] > "ü":
#         continue
     if chunk[i] not in itosSet:
        continue
     if chunk[i+1] not in itosSet:
         continue
     unigrams[chunk[i]] = unigrams.get(chunk[i], 0) + 1
     big = chunk[i:i+2]
     bigrams[big] = bigrams.get(big, 0) + 1
     counter += 1
     if counter % 100000 == 0:
       print(counter)
     if counter % 10000000 == 0:
         print("Saving")
         with open("/checkpoint/mhahn/"+args.language+"-bigrams.txt", "w") as outFile:
           for big, entry in bigrams.items():
              print(f"{big}\t{entry}", file=outFile)
#         break
#   if counter > 10000000:
#         break

from math import log

#bigrams = list(bigrams.items())
#bigrams = [(x, y, log(y)) for x, y in bigrams] #  - log(unigrams[x[0]]) - log(unigrams[x[1]])
#bigrams = sorted(bigrams, key=lambda x:x[2], reverse=True)
#print([x for x, y, z in bigrams[:100]])
#print([x for x, y, z in bigrams[-100:]])


with open("/checkpoint/mhahn/"+args.language+"-bigrams.txt", "w") as outFile:
  for big, entry in bigrams.items():
     print(f"{big}\t{entry}", file=outFile)



