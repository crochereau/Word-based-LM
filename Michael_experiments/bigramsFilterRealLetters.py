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
stoi = dict([(itos[i],i) for i in range(len(itos))])

itosSet = set(itos)

realChars = "'abcdefghijklmnopqrstuvwxyzàèéìòù"

with open("/checkpoint/mhahn/italian-bigrams.txt", "r") as inFile:
  bigrams = inFile.read().strip().split("\n")

bigramsFiltered = sorted([b for b in bigrams if b[0] in realChars and b[1] in realChars])

with open("/checkpoint/mhahn/italian-bigrams-filtered.txt", "w") as outFile:
   for bigram in bigramsFiltered:
      print(bigram, file=outFile)

quit()



