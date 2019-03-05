import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
import random

parser.add_argument("--sequence_length", type=int, default=6)

import math

args=parser.parse_args()

print(args)

import collections

queue = collections.deque(maxlen=args.sequence_length)



import corpusIteratorWiki
import random
import torch

print(torch.__version__)

from weight_drop import WeightDrop
queue.append(".")
queue.append(".")
queue.append(".")
queue.append(".")
queue.append(".")
queue.append(".")

counts = {}

data = corpusIteratorWiki.training(args.language)
 

def save():
  path = f"/checkpoint/mhahn/ngrams-{args.language}-{args.sequence_length}.txt"
  print("Saving ", path)
  with open(path, "w") as outFile:
      for gram, count in counts.items():
           if count > 5:
                print(gram, count, file=outFile)

counter = 0
for chunk in data:
       for char in chunk:
         counter += 1
         if counter % 1000000 == 0:
            print(counter)
         if char == " ":
           continue
         queue.append(char)
         gram = "".join(queue)
         counts[gram] = counts.get(gram, 0) + 1
         if counter % 1000000000 == 0:
               save()

save()
