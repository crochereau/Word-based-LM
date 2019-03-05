from paths import MODELS_HOME

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
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
training = CorpusIterator(args.language, partition="train", storeMorph=True, removePunctuation=True)
dev = CorpusIterator(args.language, partition="dev", storeMorph=True, removePunctuation=True)

def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

try:
   with open("/checkpoint/mhahn/char-vocab-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
    print("Creating new vocab")
    char_counts = {}
    # get symbol vocabulary
    for sentence in plus(training.iterator(), dev.iterator()):
       for line in sentence:
           for char in line["word"]:
              char_counts[char] = char_counts.get(char, 0) + 1
    char_counts = [(x,y) for x, y in char_counts.items()]
    itos = [x for x,y in sorted(char_counts, key=lambda z:(z[0],-z[1])) if y > 50]
    with open("/checkpoint/mhahn/char-vocab-"+args.language, "w") as outFile:
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

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      print(name)
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout

def encodeWord(word):
      numeric = [[0]]
      for char in word:
                numeric[-1].append((stoi[char]+3 if char in stoi else 2) if True else 2+random.randint(0, len(itos)))
      return numeric

rnn_drop.train(False)

def encodeSequence(numeric):
      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1).cuda(), requires_grad=False)
      embedded = char_embeddings(input_tensor)
      out, hidden = rnn_drop(embedded, None)
      return out[-1].view(-1), hidden[0].view(-1), hidden[1].view(-1)


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


plurals = set()

genders = dict([("Gender="+x, set()) for x in ["Masc", "Fem", "Neut"]])

for sentence in training.iterator():
    for line in sentence:
     if line["posUni"] == "NOUN":
      morph = line["morph"]
      if "Number=Sing" in morph:
        gender = [x for x in morph if x.startswith("Gender=")]
        if len(gender) > 0:
          genders[gender[0]].add(line["word"].lower())

      if "Number=Plur" in  morph and "Case=Dat" not in morph:
        if "|" not in line["lemma"] and line["lemma"].lower() != line["word"]:
          plurals.add((line["lemma"].lower(), line["word"]))
print(plurals)


print(genders)





