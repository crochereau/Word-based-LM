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
    devLength = int(0.05 * len(data))
    training_data = data[devLength:]
    dev = data[:devLength]

    joined = []
    for sent in data:
       joined += sent
    joined = " ".join(joined)
#    devLength = len(joined)
#    training_data = joined[10000:]
#    dev = joined[:10000]
#    data = [x.split("\t")[1].lower() for x in inFile.read().split("\n") if len(x) > 1 and not x.startswith("#")]
    #print(data)
    #data = "".join(data)
    #print(data)
itos = list(set([x for x in joined]))
itos = sorted(itos)
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])


sequenceLength = 100
batchSize = 16

#partitions = [sequenceLength*x for x in range(int(len(training_data)/sequenceLength))]



import random


import torch

print(torch.__version__)

rnn = torch.nn.LSTM(100, 1024).cuda()

output = torch.nn.Linear(1024, len(itos)+1).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+1, embedding_dim=100).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=0.05)
#dropout = torch.nn.Dropout(p=0.33)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

lr = 0.1
optim = torch.optim.SGD(parameters(), lr=lr, momentum=0.0)

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable



def forward(data, indices, train=True, printHere=False):
    
      numeric_with_blanks = [([0] + [stoi[data[x]]+1 for x in range(b, b+sequenceLength) if x < len(data)]) for b in indices]
      boundaries = [[] for _ in indices]
      numeric = [[] for _ in indices]
      assert len(numeric) > 0
      for i in range(len(indices)):
         for entry in numeric_with_blanks[i]:
           if itos[entry-1] == " ":
              boundaries[i].append(len(numeric[i]))
           else:
              numeric[i].append(entry)
      assert len(numeric[0]) > 1, numeric_with_blanks

      maxLength = max([len(x) for x in numeric])
      for i in range(len(numeric)):
        numeric[i] = numeric[i] + [0]*(maxLength-len(numeric[i]))

     # print(numeric)
      
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
#      print(numeric)
#      print(input_tensor)
#      print(input_tensor.data.size())
      target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)
      
      embedded = char_embeddings(input_tensor)
      #print(embedded.size())
      if train:
         embedded = char_dropout(embedded)

      out, _ = rnn(embedded, None)
      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)
      if printHere:
         loss = print_loss(log_probs.view(-1, len(itos)+1), target_tensor.view(-1)).view((maxLength-1), batchSize)
         losses = loss.data.cpu().numpy()
         boundaries_index = [0 for _ in range(batchSize)]
         for i in range((maxLength-1)-1):
            if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
               boundary = True
               boundaries_index[0] += 1
            else:
               boundary = False
            print((losses[i][0], itos[numeric[0][i+1]-1], boundary))
         loss = loss.mean()
      else:     
         loss = train_loss(log_probs.view(-1, len(itos)+1), target_tensor.view(-1))
      return loss

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      optim.step()
      

devLosses = []

for epoch in range(100):
   print(epoch)
   random.shuffle(training_data)
   training_join = []
   for sent in training_data:
       training_join += sent
   training_join = "".join(training_join)

#   cutoff = random.randint(0,len(training_data)-1)
#   training_data = training_data[cutoff:] + training_data[:cutoff]
   batchCounter = 0
   for batch in range(0, len(training_join), sequenceLength*batchSize):
      batchCounter += 1
      printHere = (batchCounter % 50 == 0)
       
      loss = forward(training_join, [batch+i*sequenceLength for i in range(0, batchSize)], printHere=printHere)
      backward(loss, printHere)
      if printHere:
        print(lr)
        print(devLosses)

   random.shuffle(dev)
   dev_join = []
   for sent in dev:
       dev_join += sent
   dev_join = "".join(dev_join)

#   cutoff = random.randint(0,len(dev_data)-1)
#   dev_data = dev_data[cutoff:] + dev_data[:cutoff]
   batchCounter = 0
   devLoss = 0
   devCounter = 0
   for batch in range(0, len(dev_join), sequenceLength*batchSize):
      batchCounter += 1
      printHere = (batchCounter % 50 == 0)
      loss = forward(dev_join, [batch+i*sequenceLength for i in range(0, batchSize)], printHere=printHere)
      devLoss += loss.data.cpu().numpy()
      devCounter += 1 #sequenceLength*batchSize
   devLosses.append(devLoss/devCounter)

   
   if len(devLosses) > 1 and devLosses[-1] > bestDevLoss:
     lr = 0.5 * lr
     optim = torch.optim.SGD(parameters(), lr=lr, momentum=0.0)
   else:
     bestDevLoss =  devLosses[-1]
     if args.save_to is not None:
        torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")


##      print(train[batch*batchSize:(batch+1)*batchSize])
#      numeric = [([0] + [stoi[data[x]]+1 for x in range(b, b+sequenceLength) if x < len(data)]) for b in train[batch*batchSize:(batch+1)*batchSize]]
#     # print(numeric)
#      input_tensor = Variable(torch.LongTensor(numeric[:-1]).transpose(0,1).cuda(), requires_grad=False)
#      target_tensor = Variable(torch.LongTensor(numeric[1:]).transpose(0,1).cuda(), requires_grad=False)
#
#    #  print(char_embeddings)
#      embedded = char_embeddings(input_tensor)
#      out, _ = rnn(embedded, None)
#      logits = output(out) 
#      log_probs = logsoftmax(logits)
#   #   print(logits)
#  #    print(log_probs)
# #     print(target_tensor)
#      loss = train_loss(log_probs.view(-1, len(itos)+1), target_tensor.view(-1))
#      optim.zero_grad()
#      if batch % 10 == 0:
#         print(loss)
#      loss.backward()
#      optim.step()
#      
#

