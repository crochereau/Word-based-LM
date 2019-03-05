from paths import WIKIPEDIA_HOME
from paths import LOG_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([32, 64, 128, 256]))
parser.add_argument("--char_embedding_size", type=int, default=random.choice([50, 100, 200]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024, 2048]))
parser.add_argument("--layer_num", type=int, default=random.choice([2,3]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0, 0.01, 0.05, 0.1]))
parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0, 0.01, 0.05, 0.1]))
parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.01, 0.02]))
parser.add_argument("--learning_rate", type = float, default= random.choice([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50, 80]))
parser.add_argument("--verbose", type=bool, default=False)



args=parser.parse_args()

if "MYID" in args.save_to:
   args.save_to = args.save_to.replace("MYID", str(args.myID))

print(args)



import corpusIteratorWikiShuffle



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

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout
printHere = False

def prepareDatasetChunks(data, train=True):
      global printHere
      numeric = []
      count = 0
      print("Prepare chunks")
      dataset = []
      for chunk in data:
          if len(dataset) % 500000 == 0:
              print(len(dataset))
          dataset.append(chunk)
      print("Shuffling")
      random.shuffle(dataset)
      print("Shuffled")
      startIndices = [int(i * len(dataset)/args.batchSize) for i in range(args.batchSize)]
      indicesInChunks = [0 for _ in range(args.batchSize)]
      hasNext = True
      while hasNext:         
         numeric = []
         if printHere:
            print(startIndices[0]*args.batchSize / len(dataset))
         for i in range(args.batchSize):
             result = []
             
             while len(result) < args.sequence_length:
 #                assert startIndices[i] < len(dataset), startIndices[i]
#                 assert indicesInChunks[i] < len(dataset[startIndices[i]]), dataset[startIndices[i]]
                 remainingChunk = dataset[startIndices[i]][indicesInChunks[i]:indicesInChunks[i]+args.sequence_length-len(result)]
                 indicesInChunks[i] += args.sequence_length-len(result)
                 

                 if indicesInChunks[i] >= len(dataset[startIndices[i]]):
                     startIndices[i] += 1
                     indicesInChunks[i] = 0
#                 assert startIndices[i]/len(dataset) >= i/args.batchSize, (startIndices[i]/len(dataset), i/args.batchSize)
                     if startIndices[i]/len(dataset) >= (i+1)/args.batchSize:
                         hasNext = False
                         break
                     if startIndices[i] >= len(dataset):
                         hasNext = False
                         break
                 for char in remainingChunk:
                   if char == " " or char == "\n":
                     continue
                   result.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
                   count += 1
             if len(result) == args.sequence_length:
                 assert len(result) == args.sequence_length
                 numeric.append(result)
             else:
                hasNext = False
                break
         if len(numeric) < args.batchSize /2:
             break
         if not hasNext:
            break
         yield numeric


def prepareDataset(data, train=True):
      numeric = [0]
      count = 0
      for char in data:
         if char == " ":
           continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric
            numeric = [0]


def forward(numeric, train=True, printHere=False):
      global hidden
      if True or hidden == None or (train and random.random() > 0.9): # reset the hidden state, to make sure the network is able to deal with sequences that start with 'zero'
        hidden = None
        for i in range(len(numeric)):
           numeric[i] = [0] + numeric[i]
      else:
        hidden = tuple([Variable(x.data).detach() for x in hidden])
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)


    #  print(char_embeddings)
      #if train and (embedding_full_dropout_prob is not None):
      #   embedded = embedded_dropout(char_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #char_embeddings(input_tensor)
      #else:
      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      out, hidden = rnn_drop(embedded, hidden)
#      if train:
#          out = dropout(out)

      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

      if printHere:
         lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, len(numeric))
         losses = lossTensor.data.cpu().numpy()
#         boundaries_index = [0 for _ in numeric]
         for i in range((len(numeric[0])-1)-1):
 #           if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
  #             boundary = True
   #            boundaries_index[0] += 1
    #        else:
     #          boundary = False
            print((losses[i][0], itos[numeric[0][i+1]-3]))
   #      print(lossTensor.mean())
      return loss, len(numeric) * args.sequence_length

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_cached, 5.0) #, norm_type="inf")
      optim.step()




import time

devLosses = []
for epoch in range(10000):
   print(epoch)
   training_data = corpusIteratorWikiShuffle.training(args.language)
   print("Got data")
   training_chars = prepareDataset(training_data, train=True) if args.language == "italian" else prepareDatasetChunks(training_data, train=True)



   rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden = None

   while True:
      counter += 1
      try:
         numeric = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % (50 if args.verbose else 1000) == 0)
      charCounts = 5
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
      trainChars += charCounts 
      if printHere:
          print((epoch,counter))
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))
          print(args)
      if counter % 20000 == 0 and epoch == 0:
        if args.save_to is not None:
           print("Saving")
           torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")


   rnn_drop.train(False)


   dev_data = corpusIteratorWikiShuffle.dev(args.language)
   print("Got data")
   dev_chars = prepareDataset(dev_data, train=True) if args.language == "italian" else prepareDatasetChunks(dev_data, train=True)


     
   dev_loss = 0
   dev_char_count = 0
   counter = 0
   hidden = None
   while True:
       counter += 1
       try:
          numeric = next(dev_chars)
       except StopIteration:
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
       dev_loss += numberOfCharacters * loss.cpu().data.numpy()
       dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   print(devLosses)
   with open(LOG_HOME+"/"+args.language+"_"+__file__+"_"+str(args.myID), "w") as outFile:
      print(" ".join([str(x) for x in devLosses]), file=outFile)
      print(" ".join(sys.argv), file=outFile)

   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break
   if args.save_to is not None:
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")



