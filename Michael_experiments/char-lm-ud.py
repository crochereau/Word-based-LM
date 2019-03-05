from paths import LOG_HOME
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

args=parser.parse_args()
print(args)





from corpusIterator import CorpusIterator
training = CorpusIterator(args.language, partition="train", storeMorph=False, removePunctuation=True)
dev = CorpusIterator(args.language, partition="dev", storeMorph=False, removePunctuation=True)

def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

itos = set()
# get symbol vocabulary
for sentence in plus(training.iterator(), dev.iterator()):
   for line in sentence:
       for char in line["word"]:
          itos.add(char)

itos = list(itos)
itos = sorted(itos)
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

output = torch.nn.Linear(args.hidden_dim, len(itos)+2).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+2, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequenceLength) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


def encodeUtterance(utterance, train=True):
     numeric = [0]
     boundaries = []
     for word in utterance:
        for char in word["word"]:
          numeric.append(stoi[char]+2 if random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)-1))
        boundaries.append(len(numeric))
     numeric.append(1)
     return boundaries, numeric

def forward(indices, train=True, printHere=False):
      boundaries = []
      numeric = []
      for b in indices:
         boundaries1, numeric1 = encodeUtterance(b, train)
         boundaries.append(boundaries1)
         numeric.append(numeric1)
      numberOfCharacters = sum([len(x)-1 for x in numeric]) # subtract the contribution of the initial SOS symbol
      maxLength = max([len(x) for x in numeric])
      for i in range(len(numeric)):
          numeric[i] = numeric[i] + ([0]*(maxLength - len(numeric[i])))

     # print(numeric)
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)


    #  print(char_embeddings)
      #if train and (embedding_full_dropout_prob is not None):
      #   embedded = embedded_dropout(char_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #char_embeddings(input_tensor)
      #else:
      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      out, _ = rnn_drop(embedded, None)
#      if train:
#          out = dropout(out)

      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      loss = train_loss(log_probs.view(-1, len(itos)+2), target_tensor.view(-1))

      if printHere:
         lossTensor = print_loss(log_probs.view(-1, len(itos)+2), target_tensor.view(-1)).view((maxLength-1), args.batchSize)
         losses = lossTensor.data.cpu().numpy()
         boundaries_index = [0 for _ in range(args.batchSize)]
         for i in range((maxLength-1)-1):
            if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
               boundary = True
               boundaries_index[0] += 1
            else:
               boundary = False
            print((losses[i][0], itos[numeric[0][i+1]-2], boundary))
      return loss, numberOfCharacters

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      optim.step()

import time

devLosses = []
for epoch in range(10000):
   print(epoch)
   training_data = training.iterator()
   training_data = sorted(training_data, key=lambda x:len(x))
   
   offsets = list(range(0, int(len(training_data)/args.batchSize)))
   random.shuffle(offsets)



#   oldParameters = {}
#   for name, param in rnn.named_parameters():
#      oldParameters[name] = param
#      setattr(rnn, name, torch.nn.functional.dropout(param, p=weight_dropout))
#      print(name, param.size())
#

   rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   for counter, batch in enumerate(offsets):
      printHere = (batch % 50 == 0)
      loss, charCounts = forward(training_data[batch*args.batchSize:(batch+1)*args.batchSize], printHere=printHere, train=True)
      backward(loss, printHere)
      trainChars += charCounts 
      if printHere:
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))

   rnn_drop.train(False)
#   for name, param in rnn.named_parameters():
#      setattr(rnn, name, oldParameters[name])


     
   dev_loss = 0
   dev_char_count = 0
   dev_data = dev.iterator()
   dev_data = sorted(dev_data, key=lambda x:len(x))
   for offset in range(0, len(dev_data), args.batchSize):
       printHere = (int(offset/args.batchSize) % 10 == 0)
       loss, numberOfCharacters = forward(dev_data[offset:offset+args.batchSize], printHere=printHere, train=False)
       dev_loss += numberOfCharacters * loss.cpu().data.numpy()[0]
       dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   print(devLosses)
   with open(LOG_HOME+"/"+args.language+"_"+__file__+"_"+str(args.myID), "w") as outFile:
      print(" ".join([str(x) for x in devLosses]), file=outFile)

   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break
   if args.save_to is not None:
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")



