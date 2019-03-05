from paths import MODELS_HOME

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)
args=parser.parse_args()
print(args)

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
    dev_data = joined[:100000]
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


neuron_values = [None for _ in dev_data]
boundary = [None for _ in dev_data]

for start in range(0, len(numeric_full)-sequenceLength, batchSize):
      numeric = [([0] + numeric_full[b:b+sequenceLength]) for b in range(start, start+batchSize)]
      maxLength = max([len(x) for x in numeric])
      for i in range(len(numeric)):
        numeric[i] = numeric[i] + [0]*(maxLength-len(numeric[i]))

     # print(numeric)
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
      target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)
      embedded = char_embeddings(input_tensor)


      out, hidden = rnn(embedded, None)
#      print(hidden)
#      quit()

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
            neuron_values[i+halfSequenceLength] = out[halfSequenceLength].data.cpu().numpy()[i-start]
         future_surprisal_without[i] = surprisalAtStart
             
def mi(x,y):
  return   x-y if x is not None and y is not None else None

chars = []
predictor = []
dependent = []
neuron_values_selected = []

boundaries_index = 0
for i in range(len(numeric_full)):
   if boundaries_index < len(boundaries) and i == boundaries[boundaries_index]:
      boundary = True
      boundaries_index += 1
   else:
      boundary = False
   pmiFuturePast = mi(future_surprisal_without[i], future_surprisal_with[i])
   print((itos[numeric_full[i]-1], char_surprisal[i], pmiFuturePast, pmiFuturePast < 0 if pmiFuturePast is not None else None, boundary)) # pmiFuturePast < 2 if pmiFuturePast is not None else None,
   if pmiFuturePast is not None:
     chars.append(itos[numeric_full[i]-1])
     predictor.append([pmiFuturePast, char_surprisal[i], char_entropy[i]]) #char_surprisal[i], pmiFuturePast]) #pmiFuturePast])
     dependent.append(1 if boundary else 0)
     neuron_values_selected.append([neuron_values[i-2], neuron_values[i-1], neuron_values[i], neuron_values[i+1], neuron_values[i+2]])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


for i in range(1024):
   predictor = [[y[i] if y is not None else 0 for y in x] for x in neuron_values_selected]

   logisticRegr = LogisticRegression()

   logisticRegr.fit(predictor, dependent)
   score = logisticRegr.score(predictor, dependent)
   if score > 0.8:
      print(score)




