from paths import WIKIPEDIA_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME
import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--load-from-baseline", dest="load_from_baseline", type=str)

#parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--char_embedding_size", type=int, default=200)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--layer_num", type=int, default=3)
parser.add_argument("--weight_dropout_in", type=float, default=0.01)
parser.add_argument("--weight_dropout_hidden", type=float, default=0.1)
parser.add_argument("--char_dropout_prob", type=float, default=0.33)
parser.add_argument("--char_noise_prob", type = float, default= 0.01)
parser.add_argument("--learning_rate", type = float, default= 0.1)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=50)
parser.add_argument("--printing", type=bool, default=False)
parser.add_argument("--nonlinearity", type=str, default=random.choice(["tanh", "relu"]))



parser.add_argument("--sequences", type=str)
parser.add_argument("--iterations", type=str)


import math

args=parser.parse_args()


args.sequences = args.sequences.split(",")
#args.save_to = args.save_to.replace("SEQUENCE", ".".join(args.sequences)).replace("ITERATIONS", str(args.iterations))

if args.printing:
  print(args)

args.load_from = f"wiki-{args.language}-nospaces-bptt-rnn-"+".".join(args.sequences)+"-"+str(args.iterations)



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
#itos = sorted(itos)
if args.printing:
  print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])

if args.language == "german":
  alphabet = "abcdefghijklmnopqrstuvwxyzßäöü"
elif args.language == "italian":
  alphabet = "abcdefghijklmnopqrstuvwxyzàèéìòù"
else:
   assert False

import random


import torch

if args.printing:
  print(torch.__version__)

from weight_drop import WeightDrop


rnn = torch.nn.RNN(args.char_embedding_size, args.hidden_dim, args.layer_num, args.nonlinearity).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
if args.printing:
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

if args.printing:
   print("Loading model")
if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      print(checkpoint[name].keys())
      module.load_state_dict(checkpoint[name])
else:
   assert False
####################################





from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout

def encodeWord(word):
      numeric = [[]]
      for char in word:
           numeric[-1].append((stoi[char]+3 if char in stoi else 2) if True else 2+random.randint(0, len(itos)))
      return numeric




rnn_drop.train(False)
#rnn_forward_drop.train(False)
#rnn_backward_drop.train(False)

#baseline_rnn_encoder_drop.train(False)

lossModule = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)


def choice(numeric1, numeric2):
     assert len(numeric1) == 1
     assert len(numeric2) == 1
     numeric = [numeric1[0], numeric2[0]]
     maxLength = max([len(x) for x in numeric])
     for i in range(len(numeric)):
        while len(numeric[i]) < maxLength:
              numeric[i].append(0)
     input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
     
     target = input_tensor_forward[1:]
     input_cut = input_tensor_forward[:-1]
     embedded_forward = char_embeddings(input_cut)
     out_forward, hidden_forward = rnn_drop(embedded_forward, None)

     prediction = logsoftmax(output(out_forward)) #.data.cpu().view(-1, 3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
     losses = lossModule(prediction.view(-1, len(itos)+3), target.view(-1)).view(maxLength, 2)
     losses = losses.sum(0).data.cpu().numpy()
     return losses



def choiceList(numeric):
     for x in numeric:
       assert len(x) == 1
#     assert len(numeric1) == 1
 #    assert len(numeric2) == 1
     numeric = [x[0] for x in numeric] #, numeric2[0]]
     maxLength = max([len(x) for x in numeric])
     for i in range(len(numeric)):
        while len(numeric[i]) < maxLength:
              numeric[i].append(0)
     input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
     
     target = input_tensor_forward[1:]
     input_cut = input_tensor_forward[:-1]
     embedded_forward = char_embeddings(input_cut)
     out_forward, hidden_forward = rnn_drop(embedded_forward, None)

     prediction = logsoftmax(output(out_forward)) #.data.cpu().view(-1, 3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
     losses = lossModule(prediction.view(-1, len(itos)+3), target.view(-1)).view(maxLength, len(numeric))
     losses = losses.sum(0).data.cpu().numpy()
     return losses



def encodeSequenceBatchForward(numeric):
      input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)

#      target_tensor_forward = Variable(torch.LongTensor(numeric).transpose(0,1)[2:].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
      embedded_forward = char_embeddings(input_tensor_forward)
      out_forward, hidden_forward = rnn_drop(embedded_forward, None)
#      out_forward = out_forward.view(args.sequence_length+1, len(numeric), -1)
 #     logits_forward = output(out_forward) 
  #    log_probs_forward = logsoftmax(logits_forward)
      return (out_forward[-1], hidden_forward)



def encodeSequenceBatchBackward(numeric):
#      print([itos[x-3] for x in numeric[0]])
#      print([[0]+(x[::-1]) for x in numeric])
      input_tensor_backward = Variable(torch.LongTensor([[0]+(x[::-1]) for x in numeric]).transpose(0,1).cuda(), requires_grad=False)
#      target_tensor_backward = Variable(torch.LongTensor([x[::-1] for x in numeric]).transpose(0,1)[:-2].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
      embedded_backward = char_embeddings(input_tensor_backward)
      out_backward, hidden_backward = rnn_backward_drop(embedded_backward, None)
#      out_backward = out_backward.view(args.sequence_length+1, len(numeric), -1)
#      logits_backward = output(out_backward) 
#      log_probs_backward = logsoftmax(logits_backward)

      return (out_backward[-1], hidden_backward)


import numpy as np

def predictNext(encoded, preventBoundary=True):
     out, hidden = encoded
     prediction = logsoftmax(output(out.unsqueeze(0))).data.cpu().view(3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
     predicted = np.argmax(prediction[:-1] if preventBoundary else prediction)
     return itos[predicted-3] #, prediction

def keepGenerating(encoded, length=100, backwards=False):
    out, hidden = encoded
    output_string = ""
   
#    rnn_forward_drop.train(True)

    for _ in range(length):
      prediction = logsoftmax(2*output(out.unsqueeze(0))).data.cpu().view(3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
#      predicted = np.argmax(prediction).items()
      predicted = np.random.choice(3+len(itos), p=np.exp(prediction))

      output_string += itos[predicted-3]

      input_tensor_forward = Variable(torch.LongTensor([[predicted]]).transpose(0,1).cuda(), requires_grad=False)

      embedded_forward = char_embeddings(input_tensor_forward)
      
      out, hidden = (rnn_drop if not backwards else rnn_backward_drop)(embedded_forward, hidden)
      out = out[-1]

 #   rnn_forward_drop.train(False)


    return output_string if not backwards else output_string[::-1]


out1, hidden1 = encodeSequenceBatchForward(encodeWord("katze"))
out2, hidden2 = encodeSequenceBatchForward(encodeWord("katzem"))
#print(torch.dot(out1[-1], out2[-1]))
#print(torch.dot(hidden1[0], hidden2[0]))
#print(torch.dot(hidden1[1], hidden2[1]))

if args.printing:
   print(torch.nn.functional.cosine_similarity(out1, out2, dim=0))
#print(torch.nn.functional.cosine_similarity(hidden1, hidden2, dim=0))
#print(torch.nn.functional.cosine_similarity(cell1, cell2, dim=0))

#print("willmach")
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ichmach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".dumach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ermach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".siemach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".esmach"))))
#
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ichmach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".dumach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ermach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".siemach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".esmach"))))
#print(keepGenerating(encodeSequenceBatchForward(encodeWord(".esdenk"))))
#
def doChoiceList(xs, printHere=False, printRanking=False):
    if printHere:
      for x in xs:
         print(x)
    losses = choiceList([encodeWord(x) for x in xs]) #, encodeWord(y))
    if printHere:
      print(losses)
    if printRanking:
       z = zip(xs, losses)
       z= sorted(z, key=lambda x:x[1])
       print(z)
    return np.argmin(losses)

def doChoiceListRet(xs, printHere=True, printRanking=False):
    if printHere:
      for x in xs:
         print(x)
    losses = choiceList([encodeWord(x) for x in xs]) #, encodeWord(y))
    if printHere:
      print(losses)
    if printRanking:
       z = zip(xs, losses)
       z= sorted(z, key=lambda x:x[1])
       print(z)
    return losses


def doChoice(x, y):
    print(x)
    print(y)
    losses = choice(encodeWord(x), encodeWord(y))
    print(losses)
    return 0 if losses[0] < losses[1] else 1


def keepGenerating(encoded, length=100, backwards=False):
    out, hidden = encoded
    output_string = ""
   
#    rnn_forward_drop.train(True)

    for _ in range(length):
      prediction = logsoftmax(2*output(out.unsqueeze(0))).data.cpu().view(3+len(itos)).numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
#      predicted = np.argmax(prediction).items()
      predicted = np.random.choice(3+len(itos), p=np.exp(prediction))

      output_string += itos[predicted-3]

      input_tensor_forward = Variable(torch.LongTensor([[predicted]]).transpose(0,1).cuda(), requires_grad=False)

      embedded_forward = char_embeddings(input_tensor_forward)
      
      out, hidden = (rnn_drop if not backwards else rnn_backward_drop)(embedded_forward, hidden)
      out = out[-1]

 #   rnn_forward_drop.train(False)


    return output_string if not backwards else output_string[::-1]

if args.printing:
   print(keepGenerating(encodeSequenceBatchForward(encodeWord(".ichmach"))))


#
#doChoice(".ichmachedas.", ".ichmachstdas.")
#doChoice(".dumachendas.", ".dumachstdas.")
#doChoice(".ermachendas.", ".ermachtdas.")
#doChoice(".wirmachendas.", ".wirmachtdas.")
#
#doChoice(".ichvergeigedas.", ".ichvergeigstdas.")
#doChoice(".duvergeigendas.", ".duvergeigstdas.")
#doChoice(".ervergeigendas.", ".ervergeigtdas.")
#doChoice(".wirvergeigendas.", ".wirvergeigtdas.")
#
#
#
#
#
#doChoice(".ichwilldas.", ".ichwillstdas.")
#doChoice(".duwollendas.", ".duwillstdas.")
#doChoice(".erwollendas.", ".erwilldas.")
#doChoice(".wirwollendas.", ".wirwilldas.")
#
#
#doChoice("indashaus.", "indiehaus.")
#doChoice("indascomputermaus.", "indiecomputermaus.")
#
#doChoice(".ichgeheindashaus.", ".ichgeheindemhaus.")
#doChoice(".ichlebeindashaus.", ".ichlebeindemhaus.")
#
#
#doChoice(".ichlebeindashausmeisterzimmer.", ".ichlebeindemhausmeisterzimmer.")
#
#
#doChoice(".zweihaus.", ".zweihäuser.")
#doChoice(".zweilampen.", ".zweilampe.")
#doChoice(".zweilampenpfahl.", ".zweilampenpfähle.")
#doChoice(".zweihauspfähle.", ".zweihäuserpfähle.")
#doChoice(".zweinasenbär.", ".zweinasenbären.")
#
#doChoice(".einhaus.", ".einhäuser.")
#doChoice(".einlampenpfahl.", ".einlampenpfähle.")
#doChoice(".einhauspfähle.", ".einhäuserpfähle.")
#doChoice(".einnasenbär.", ".einnasenbären.")


# need to do some sort of ngram control

examples = ["."+args.sequences[0][:-1]+x for x in [y[-1] for y in args.sequences]]
comb = doChoiceListRet(examples) # if we replace "sind" with "ist", the pattern changes massively (these 'odd' words are apparently preferred to be neuters)
print("===========================")
print(list(zip(examples, comb)))
print(comb[0]-comb[1])
print(math.exp(comb[0]-comb[1]))
print(math.exp(comb[1]-comb[0]))
print(">>>>",examples[0], " ", examples[1], " ",round(comb[1]-comb[0], 2))




quit()

examples = [x for x in itos[3:] if x > "_"]
uni = doChoiceListRet(examples) # if we replace "sind" with "ist", the pattern changes massively (these 'odd' words are apparently preferred to be neuters)


z = zip(examples, comb, uni)
z = sorted(z, key=lambda x:x[1]-x[2])
print(z)

quit()



with open(f"/checkpoint/mhahn/CHAR_SEQUENCES_wiki-{args.language}-nospaces-bptt-"+".".join(args.sequences)+"-"+str(args.iterations), "r") as inFile:
    trigrams =[y.split("\t") for y in inFile.read().strip().split("\n")]
    
    trigrams = dict([(lambda x: (x[0], int(x[1])))(y) for y in trigrams])

bigramsLeft   = dict([(x[0:2], 0) for x in trigrams])
bigramsRight  = dict([(x[1:3], 0) for x in trigrams])
unigramsMid   = dict([(x[1], 0) for x in trigrams])
unigramsRight = dict([(x[2], 0) for x in trigrams])

total = len(itos)
for t, count in trigrams.items():
   total += count
   bigramsLeft[t[0:2]] += count
   bigramsRight[t[1:3]] += count
   unigramsMid[t[1]] += count
   unigramsRight[t[2]] += count

from math import log

def evaluateNgramProb(x):
   
   prob = log(unigramsRight.get(x[0], 0) + 1) - log(total)
   if x[0:2] in bigramsRight:
      prob += log(bigramsRight[x[0:2]]) - log(unigramsMid[x[0]])
   elif x[1] in unigramsRight:
       prob = log(1+unigramsRight[x[1]]) - log(total)
   else:
       prob -= log(total)
   for i in range(2, len(x)):
      if x[i-2:i+1] in trigrams:
           prob += log(trigrams[x[i-2:i+1]]) - log(bigramsLeft[x[i-2:i]])
      elif x[i-1:i+1] in bigramsRight:
          prob += log(bigramsRight[x[i-1:i+1]]) - log(unigramsMid[x[i-1]])
      elif x[i] in unigramsRight:
          prob += log(1+unigramsRight[x[i]]) - log(total)
      else:
          prob -= log(total)
   return prob

def doChoiceListNgrams(xs):
    return [-evaluateNgramProb(x) for x in xs]


quit()


examples = ["."+args.sequences[0][:-1]+x for x in alphabet]
comb = doChoiceListNgrams(examples) # if we replace "sind" with "ist", the pattern changes massively (these 'odd' words are apparently preferred to be neuters)
examples = [x for x in itos[3:] if x > "_"]
uni = doChoiceListNgrams(examples) # if we replace "sind" with "ist", the pattern changes massively (these 'odd' words are apparently preferred to be neuters)


z = zip(examples, comb, uni)
z = sorted(z, key=lambda x:x[1]-x[2])
print(z)



