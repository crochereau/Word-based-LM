from paths import MODELS_HOME

# final product from Friday


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)
args=parser.parse_args()
print(args)

import math

import random

with open("/private/home/mhahn/data/br-data/br-phono.txt", "r") as inFile:
    data = inFile.read().split("\n")
    random.Random(6598).shuffle(data) # now the sentences are arranged in random order

#    print([[y.split("\t")[1].lower() for y in x.split("\n") if len(y) > 1 and y[0][0] is not "#"] for x in data])

    data = [y.split(" ") for y in data]
    joined = []
    for sent in data:
       joined += sent
    joined = " ".join(joined)
    training_data = joined[10000:]
    dev_data = joined #[:10000]
#    data = [x.split("\t")[1].lower() for x in inFile.read().split("\n") if len(x) > 1 and not x.startswith("#")]
    #print(data)
    #data = "".join(data)
    #print(data)
itos = list(set([x for x in joined]))
itos = sorted(itos)
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])

halfSequenceLength = 5
sequenceLength = 2*halfSequenceLength
batchSize = 16

partitions = [sequenceLength*x for x in range(int(len(dev_data)/sequenceLength))]



import random


import torch

print(torch.__version__)

rnn = torch.nn.LSTM(100, 1024).cuda()

output = torch.nn.Linear(1024, len(itos)+1).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+1, embedding_dim=100).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=0.33)
#dropout = torch.nn.Dropout(p=0.33)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

optim = torch.optim.SGD(parameters(), lr=0.1, momentum=0.0)

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


future_surprisal_with = [None for _ in dev_data]
future_surprisal_without = [None for _ in dev_data]

char_surprisal = [None for _ in dev_data]
char_entropy = [None for _ in dev_data]

numeric_with_blanks = [stoi[x]+1 for x in dev_data]

boundaries = []
numeric_full = []
for entry in numeric_with_blanks:
  if itos[entry-1] == " ":
     boundaries.append(len(numeric_full))
  else:
     numeric_full.append(entry)



for start in range(0, len(numeric_full)-sequenceLength, batchSize):
      numeric = [([0] + numeric_full[b:b+sequenceLength]) for b in range(start, start+batchSize)]
      maxLength = max([len(x) for x in numeric])
      for i in range(len(numeric)):
        numeric[i] = numeric[i] + [0]*(maxLength-len(numeric[i]))

     # print(numeric)
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)
      embedded = char_embeddings(input_tensor)


      out, _ = rnn(embedded, None)
      logits = output(out) 
      log_probs = logsoftmax(logits)

      entropy = (- log_probs * torch.exp(log_probs)).sum(2).view((maxLength-1), batchSize).data.cpu().numpy()

      loss = print_loss(log_probs.view(-1, len(itos)+1), target_tensor.view(-1)).view((maxLength-1), batchSize)
      losses = loss.data.cpu().numpy()
#      for i in range(len(numeric[0])-1):
#         print((i,losses[i][0], itos[numeric[0][i+1]-1]))
      for i in range(start, start+batchSize):
         #print(losses[:int(halfSequenceLength),i-start].sum())
         surprisalAtStart = losses[:halfSequenceLength,i-start].sum()
         surprisalAtMid = losses[halfSequenceLength:, i-start].sum()
         #print(losses[:,i-start])
         if i+halfSequenceLength < len(future_surprisal_with):
            future_surprisal_with[i+halfSequenceLength] = surprisalAtMid
            char_surprisal[i+halfSequenceLength] = losses[halfSequenceLength, i-start]
            char_entropy[i+halfSequenceLength] = entropy[halfSequenceLength, i-start]
         future_surprisal_without[i] = surprisalAtStart
             
def mi(x,y):
  return   x-y if x is not None and y is not None else None

characters = []
predictor = []
dependent = []

pmiFuturePast = [None for _ in numeric_full]

isBoundary = [None for _ in numeric_full]

characters = [None for _ in numeric_full]

boundaries_index = 0
for i in range(len(numeric_full)):
   if boundaries_index < len(boundaries) and i == boundaries[boundaries_index]:
      boundary = True
      boundaries_index += 1
   else:
      boundary = False
   pmiFuturePast[i] = mi(future_surprisal_without[i], future_surprisal_with[i])
#   print((itos[numeric_full[i]-1], char_surprisal[i], pmiFuturePast, pmiFuturePast < 0 if pmiFuturePast is not None else None, boundary)) # pmiFuturePast < 2 if pmiFuturePast is not None else None,
#   if pmiFuturePast is not None:
   characters[i] = itos[numeric_full[i]-1]
#     predictor.append([pmiFuturePast, char_surprisal[i], char_entropy[i]]) #char_surprisal[i], pmiFuturePast]) #pmiFuturePast])
   isBoundary[i]= (1 if boundary else 0)


# , char_surprisal[i], char_entropy[i]

# char_surprisal[i], 

#print(predictor)
#print(dependent)


pmiFuturePast = [pmiFuturePast[i] - 0.5*(pmiFuturePast[i-1] + pmiFuturePast[i+1]) if i > 0 and i+1 < len(pmiFuturePast) and None not in pmiFuturePast[i-1:i+2] else 0 for i in range(len(pmiFuturePast))]
currentBoundaries = [x < 1.0 if x is not None else True for x in pmiFuturePast ]
currentBoundaries[0] = True
currentBoundaries[-1] = True

lexicon = {}
currentWord = ""
miPastFuture = [0.0, 0]
for i in range(len(numeric_full)):
   if currentBoundaries[i]:
      lexicon[currentWord] = lexicon.get(currentWord, 0) + 1
#      print(pmiFuturePast[i])
      if pmiFuturePast[i] is not None:
        miPastFuture[0] += pmiFuturePast[i]
        miPastFuture[1] += 1
      currentWord = ""
   currentWord += characters[i]

wordLengthTotal = sum([len(x) for x in lexicon])

alpha = -1000.001
beta = 0.02

#print(pmiFuturePast)
#print(currentBoundaries)
#quit()
#
sweep = 0
while True:
    sweep += 1
    if sweep > 2000:
      break
#    print("LEXICON",[(x,y) for x, y in lexicon.items() if y > 0])
    print (sweep, miPastFuture[0]/miPastFuture[1], wordLengthTotal, beta * wordLengthTotal)
    print(miPastFuture[0]/miPastFuture[1] + beta * wordLengthTotal)
    print("Border recall", sum([x and (y == 1) for x, y in zip(currentBoundaries, isBoundary)])/sum(isBoundary))
    print("Border precision", sum([x and (y == 1) for x, y in zip(currentBoundaries, isBoundary)])/sum([1 if x else 0 for x in currentBoundaries]))

    quality = miPastFuture[0]/miPastFuture[1] + beta * wordLengthTotal
    for index in range(1, len(numeric_full)-1):
       if pmiFuturePast[index] is None:
           continue
#       index = random.randint(1, len(numeric_full)-2)
       # add a boundary here

       left = index-1
       while not currentBoundaries[left]:
           left -= 1
       right = index+1
       while not currentBoundaries[right]:
           right += 1
#       print(characters[:50])

 #      print(index, currentBoundaries[index], left, right, characters[left:index], characters[index:right])
       if not currentBoundaries[index]:
            # add boundary here
            # find left boundary
            currentWord = "".join(characters[left:right])
            assert currentWord in lexicon
            newWord1 = "".join(characters[left:index])
            newWord2 = "".join(characters[index:right])

            newWordLengthTotal = wordLengthTotal
            if lexicon[currentWord] == 1:
                 newWordLengthTotal -= len(currentWord)
            if lexicon.get(newWord1, 0) == 0:
                 newWordLengthTotal += len(newWord1)
            if lexicon.get(newWord2, 0) == 0:
                 newWordLengthTotal += len(newWord2)
            newMiPastFuture = [miPastFuture[0] + pmiFuturePast[index], miPastFuture[1] + 1]
            newQuality = newMiPastFuture[0]/newMiPastFuture[1] + beta * newWordLengthTotal
            acceptanceProb = (1/(1 + math.exp( alpha * (quality - newQuality))))
            assert acceptanceProb >= 0.0, acceptanceProb
            assert acceptanceProb <= 1.0, acceptanceProb
            acceptChange = (random.random() < acceptanceProb)
            if random.random() < 0.00002:
               print(acceptanceProb)
#            print(acceptChange)
            #print (newQuality, quality, 1/(1 + math.exp( alpha * (quality - newQuality))), acceptChange, miPastFuture[0]/miPastFuture[1], wordLengthTotal, sweep)
            if acceptChange:
                #print("ADD")
                currentBoundaries[index] = True
                lexicon[currentWord] = lexicon[currentWord] - 1
                lexicon[newWord1] = lexicon.get(newWord1, 0) + 1
                lexicon[newWord2] = lexicon.get(newWord2, 0) + 1
                wordLengthTotal = newWordLengthTotal
                miPastFuture = newMiPastFuture
                quality = newQuality
       else:
           assert currentBoundaries[index]
           wordLeft = "".join(characters[left:index])
           wordRight = "".join(characters[index:right])
           newWord = "".join(characters[left:right])
           
           newWordLengthTotal = wordLengthTotal
           assert lexicon[wordLeft] > 0, wordLeft
           assert lexicon[wordRight] > 0, wordRight
           if lexicon[wordLeft] == 1:
                 newWordLengthTotal -= len(wordLeft)
           if lexicon[wordRight] == 1:
                 newWordLengthTotal -= len(wordRight)
           if lexicon.get(newWord, 0) == 0:
                newWordLengthTotal += len(newWord)
           newMiPastFuture = [miPastFuture[0] - pmiFuturePast[index], miPastFuture[1] - 1]
           newQuality = newMiPastFuture[0]/newMiPastFuture[1] + beta * newWordLengthTotal
           acceptChange = (random.random() < 1/(1 - math.exp( alpha * (quality - newQuality))))
           #print (newQuality, quality, 1/(1 + math.exp( alpha * (quality - newQuality))), acceptChange, miPastFuture[0]/miPastFuture[1], wordLengthTotal, sweep)
           if acceptChange:
                #print("REMOVE")
                currentBoundaries[index] = False
                lexicon[wordLeft] = lexicon[wordLeft] - 1
                lexicon[wordRight] = lexicon[wordRight] - 1
                #print((wordLeft, lexicon[wordLeft]))
                lexicon[newWord] = lexicon.get(newWord, 0) + 1
                wordLengthTotal = newWordLengthTotal
                miPastFuture = newMiPastFuture
                quality = newQuality
           


