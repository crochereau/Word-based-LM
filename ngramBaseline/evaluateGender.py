
import os
import sys
import random

# Select all files in stimuli folder starting with german-gender and ending with _counts.txt
files = [x for x in os.listdir("../stimuli/") if x.endswith("_counts.txt") and x.startswith("german-gender")]
# for each file in this list
for fileName in files:
    # initialization of counts
   counts = [0, 0, 0]

   # opens each file
   with open("../stimuli/"+fileName, "r") as inFile:
      # removes all whitespaces from the stimuli and aligns them in column
      data = inFile.read().strip().split("\n")

   for i in range(0, len(data), 3):
     countsHere = [int(x.split("\t")[1]) for x in data[i:i+3]]
     maximal = max(countsHere)
     argmax = [x for x in range(3) if countsHere[x] == maximal]
     chosen= argmax[random.randint(0, len(argmax)-1)]
#     if maximal == 0:
#        print(chosen)
     counts[chosen] += 1
   print(fileName)
#   print(i)
 #  print(counts)
   print([round(float(x)/sum(counts),2) for x in counts])

