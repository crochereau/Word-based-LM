from paths import WIKIPEDIA_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--load-from-baseline", dest="load_from_baseline", type=str)

#parser.add_argument("--save-to", dest="save_to", type=str)

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





from corpusIterator import CorpusIterator
training = CorpusIterator("German", partition="train", storeMorph=True, removePunctuation=True)
dev = CorpusIterator("German", partition="dev", storeMorph=True, removePunctuation=True)

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

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

print("Loading model")
if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      print(checkpoint[name].keys())
      module.load_state_dict(checkpoint[name])
else:
   assert False
####################################

#baseline_rnn_encoder = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num, bidirectional=True).cuda()
#baseline_rnn_encoder_drop = WeightDrop(baseline_rnn_encoder, [(name, args.weight_dropout_in) for name, _ in baseline_rnn_encoder.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in baseline_rnn_encoder.named_parameters() if name.startswith("weight_hh_")])
#
#baseline_char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()


#
#baseline_rnn_parameter_names = [name for name, _ in baseline_rnn_encoder.named_parameters()]
#print(baseline_rnn_parameter_names)
##quit()
#
#baseline_output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()
#
#baseline_modules = [baseline_rnn_encoder, baseline_output, baseline_char_embeddings]
#def baseline_parameters():
#   for module in baseline_modules:
#       for param in module.parameters():
#            yield param
#
#optim = torch.optim.SGD(baseline_parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9
#
#baseline_named_modules = {"rnn_encoder" : baseline_rnn_encoder,  "output" : baseline_output, "char_embeddings" : baseline_char_embeddings}
#
#if args.load_from_baseline is not None:
#  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from_baseline+".pth.tar")
#  for name, module in baseline_named_modules.items():
#      module.load_state_dict(checkpoint[name])
#else:
#   assert False








from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout

def encodeWord(word):
      numeric = [[0]]
      for char in word:
                numeric[-1].append((stoi[char]+3 if char in stoi else 2) if True else 2+random.randint(0, len(itos)))
      return numeric

def encodeWordBidirectional(word):
      numeric = [[0,0]]
      for char in word:
                numeric[-1].append((stoi[char]+3 if char in stoi else 2) if True else 2+random.randint(0, len(itos)))
      numeric[-1].append(0)
      numeric[-1].append(0)

      return numeric







def encodeSequenceBaseline(numeric):
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:-1].cuda(), requires_grad=False)
      target_tensor_forward = Variable(torch.LongTensor(numeric).transpose(0,1)[2:].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
      target_tensor_backward = Variable(torch.LongTensor(numeric).transpose(0,1)[:-2].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
      target_tensor = torch.cat([target_tensor_forward, target_tensor_backward], dim=2)

      embedded = baseline_char_embeddings(input_tensor)

      out, encoded = baseline_rnn_encoder_drop(embedded, None)
      out = out.view(args.sequence_length+1, len(numeric), 2, -1)
      out1 = out[-1, 0, 0, :] # forward encoding
      out2 = out[0, -1, 1, :] # backward encoding
      return out1.view(-1), out2.view(-1), encoded[0].view(-1), encoded[1].view(-1)

rnn_drop.train(False)

def encodeSequence(numeric):
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1).cuda(), requires_grad=False)
      embedded = char_embeddings(input_tensor)
      out, hidden = rnn_drop(embedded, None)
      return out[-1].view(-1), hidden[0].view(-1), hidden[1].view(-1)

def encodeSequenceBatch(numeric):
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1).cuda(), requires_grad=False)
      embedded = char_embeddings(input_tensor)
      out, hidden = rnn_drop(embedded, None)
      return out[-1], hidden[0][0], hidden[1][0]

def encodeBaselineSequenceBatch(numeric):
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:-1].cuda(), requires_grad=False)
#      target_tensor_forward = Variable(torch.LongTensor(numeric).transpose(0,1)[2:].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
#      target_tensor_backward = Variable(torch.LongTensor(numeric).transpose(0,1)[:-2].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
#      target_tensor = torch.cat([target_tensor_forward, target_tensor_backward], dim=2)
#
      embedded = baseline_char_embeddings(input_tensor)

      out, encoded = baseline_rnn_encoder_drop(embedded, None)
      out = out.view(len(numeric[0])-2, len(numeric), 2, -1)
      out1 = out[-1, 0, 0, :] # forward encoding
      assert False
      out2 = out[0, -1, 1, :] # backward encoding
      return out1.view(-1), out2.view(-1), encoded[0].view(-1) #, encoded[1].view(-1)


import numpy as np

def predictNext(numeric):
     out, hidden, cell = encodeSequence(numeric)
     prediction = logsoftmax(output(out.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
     predicted = np.argmax(prediction)
     return itos[predicted-3] #, prediction

out1, hidden1, cell1 = encodeSequence(encodeWord("katze"))
out2, hidden2, cell2 = encodeSequence(encodeWord("katzem"))
#print(torch.dot(out1[-1], out2[-1]))
#print(torch.dot(hidden1[0], hidden2[0]))
#print(torch.dot(hidden1[1], hidden2[1]))

print(torch.nn.functional.cosine_similarity(out1, out2, dim=0))
print(torch.nn.functional.cosine_similarity(hidden1, hidden2, dim=0))
print(torch.nn.functional.cosine_similarity(cell1, cell2, dim=0))

print(predictNext(encodeWord("willmach")))
print(predictNext(encodeWord("habegemach")))
print(predictNext(encodeWord("geseh")))
print(predictNext(encodeWord("gedach")))
print(predictNext(encodeWord("habegepups")))
print(predictNext(encodeWord("habegegurk")))
print(predictNext(encodeWord("habegerief")))
print(predictNext(encodeWord("ichwerdepups")))

print(predictNext(encodeWord("einekatze")))
print(predictNext(encodeWord("zweikatze")))
print(predictNext(encodeWord("derkater")))
print(predictNext(encodeWord("deskater")))

def padWords(words):
   maxLength = max([len(x) for x in words])
   for i, word in enumerate(words):
      if len(word) < maxLength:
         words[i] = ([0] * (maxLength - len(word))) + word #word.append(0)
   return words


# TODO train ght eRNN so that it is actually used to seeing 0's in the beginning of a sequence

def getEncodingsForList(wordsToBeEncoded):
    return getEncodingsForListGeneral(wordsToBeEncoded, encodeSequenceBatch)


def getEncodingsForListBaseline(wordsToBeEncoded):
    return getEncodingsForListGeneral(wordsToBeEncoded, encodeBaselineSequenceBatch)



def getEncodingsForListGeneral(wordsToBeEncoded, encodingFunction):
    modelVectors = []
    byLength = sorted(list(wordsToBeEncoded), reverse=True)

    for offset in range(0, len(wordsToBeEncoded), 100):
#      print(offset)
      codes1, codes2, codes3 = encodingFunction(padWords([encodeWord(word)[0] for word in byLength[offset:offset+100]]))
      for index, word in enumerate(byLength[offset:offset+100]):
         code1 = codes1[index].cpu()#,len(word)]
         code2 = codes2[index].cpu()#,len(word)]
         code3 = codes3[index].cpu()#,len(word)]
         modelVectors.append((code1, code2, code3))
    #     print((code1,code2,code3))
    return modelVectors


wordFrequencies = {}
for sentence in training.iterator():
   for line in sentence:
      wordFrequencies[line["word"]] = wordFrequencies.get(line["word"], 0) + 1
wordFrequencies = sorted(list(wordFrequencies.items()),key=lambda x:x[1], reverse=True)


first_n = 50
wordFrequencies = wordFrequencies[100:]
frequentWords = [x for x, y in wordFrequencies[:first_n]]

import numpy as np

encoded = np.array([x[0].cpu().data.numpy() for x in getEncodingsForList(frequentWords)])
print("Collected words")

import torch
import tsne

import json

import sys


import pylab

import matplotlib.pyplot as plt
plt.switch_backend('agg')

X = encoded
print(X)
Y = tsne.tsne(X=X, no_dims=2, initial_dims=10, perplexity=30.0)
labels = frequentWords
fig, ax = plt.subplots()
for i in range(first_n):
   ax.annotate(frequentWords[i], (Y[i,0], Y[i,1]))
ax.scatter(Y[:, 0], Y[:, 1], 20, [5.0 for _ in range(first_n)])

plt.show()
plt.savefig("t-sne-medium.png") 
        #pylab.scatter(Y[:, 0], Y[:, 1], 20, [5.0 for _ in range(50)])
        #pylab.show()




