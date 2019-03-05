from paths import MODELS_HOME

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--load-from-baseline", dest="load_from_baseline", type=str)

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
else:
   assert False
####################################

baseline_rnn_encoder = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num, bidirectional=True).cuda()
baseline_rnn_encoder_drop = WeightDrop(baseline_rnn_encoder, [(name, args.weight_dropout_in) for name, _ in baseline_rnn_encoder.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in baseline_rnn_encoder.named_parameters() if name.startswith("weight_hh_")])

baseline_char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()



baseline_rnn_parameter_names = [name for name, _ in baseline_rnn_encoder.named_parameters()]
print(baseline_rnn_parameter_names)
#quit()

baseline_output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

baseline_modules = [baseline_rnn_encoder, baseline_output, baseline_char_embeddings]
def baseline_parameters():
   for module in baseline_modules:
       for param in module.parameters():
            yield param

optim = torch.optim.SGD(baseline_parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

baseline_named_modules = {"rnn_encoder" : baseline_rnn_encoder,  "output" : baseline_output, "char_embeddings" : baseline_char_embeddings}

if args.load_from_baseline is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from_baseline+".pth.tar")
  for name, module in baseline_named_modules.items():
      module.load_state_dict(checkpoint[name])
else:
   assert False








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
      assert False
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
baseline_rnn_encoder_drop.train(False)

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

      embedded = baseline_char_embeddings(input_tensor)

      out, encoded = baseline_rnn_encoder_drop(embedded, None)
      out = out.view(len(numeric[0])-2, len(numeric), 2, -1)
 #     print(out)
      out1 = out[-1, :, 0, :] # forward encoding
      out2 = out[0, :, 1, :] # backward encoding
      out3 = encoded[0][0,:, :]
      out4 = encoded[1][0, :, :]
      out5 = encoded[0][1, :, :]
      out6 = encoded[1][1, :, :]
#      print(out1)
 #     print(len(numeric))
      out1 = torch.cat([out3, out4, out5, out6], dim=1)
      return out1, out2, out3 #, encoded[1].view(-1)










def encodeSequenceBaseline(numeric):
      assert False
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
baseline_rnn_encoder_drop.train(False)

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

      embedded = baseline_char_embeddings(input_tensor)

      out, encoded = baseline_rnn_encoder_drop(embedded, None)
      out = out.view(len(numeric[0])-2, len(numeric), 2, -1)
 #     print(out)
      out1 = out[-1, :, 0, :] # forward encoding
      out2 = out[0, :, 1, :] # backward encoding
      out3 = encoded[0][0,:, :]
      out4 = encoded[1][0, :, :]
      out5 = encoded[0][1, :, :]
      out6 = encoded[1][1, :, :]
#      print(out1)
 #     print(len(numeric))
      out1 = torch.cat([out3, out4, out5, out6], dim=1)
      return out1, out2, out3 #, encoded[1].view(-1)


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
    #     print("CODES")
    #     print(code1)
         code2 = codes2[index].cpu()#,len(word)]
         code3 = codes3[index].cpu()#,len(word)]
         modelVectors.append((code1, code2, code3))
    #     print((code1,code2,code3))
    return modelVectors





plurals = set()

genders = dict([("Gender="+x, set()) for x in ["Masc", "Fem", "Neut"]])

for sentence in training.iterator():
    for line in sentence:
     if line["posUni"] == "NOUN":
      morph = line["morph"]
      if "Number=Sing" in morph and "Case=Nom" in morph:
#        if not line["word"].endswith("e"):
          gender = [x for x in morph if x.startswith("Gender=")]
          if len(gender) > 0:
            genders[gender[0]].add(line["word"].lower())

      if "Number=Plur" in  morph and "Case=Dat" not in morph:
        if "|" not in line["lemma"] and line["lemma"].lower() != line["word"]:
          plurals.add((line["lemma"].lower(), line["word"]))
print(plurals)


#pluralWords = []
#singularWords = []
#for word in plurals:
#   singularWords.append(word[0])
#   pluralWords.append(word[1])
#
#plur = getEncodingsForList(pluralWords)
#sing = getEncodingsForList(singularWords)
#
#
#print("Concatenating")
#
#predictors = []
#dependent = []
#for vectors in plus(sing, plur):
#     code = vectors[0] #torch.cat(vectors, dim=0)
#    # print(code)
#     predictors.append(code.data.cpu().numpy())
#for _ in sing:
#  dependent.append(0)
#for _ in plur:
#  dependent.append(1)
# 
#
## create logistic regression for gender
#
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(predictors, dependent, test_size=0.1, random_state=0, shuffle=True)
#
#
#from sklearn.linear_model import LogisticRegression
#
#print("regression")
#
#logisticRegr = LogisticRegression()
#
#logisticRegr.fit(x_train, y_train)
#
#predictions = logisticRegr.predict(x_test)
#
#
#score = logisticRegr.score(x_test, y_test)
#print(score)
#




######################################

print(genders)

# create a dictionary of encodings of all words

# then see whether things are more predictable from LM than from baseline

wordsToBeEncoded = genders["Gender=Neut"]

baselineVectors = []

print(len(genders["Gender=Fem"]))
print(len(genders["Gender=Masc"]))

femWords = random.sample(genders["Gender=Fem"], 1000)
mascWords = random.sample(genders["Gender=Masc"], 1000)


## sanity check (currently failing)
#words = femWords+mascWords
#random.shuffle(words)
#femWords = words[:1000]
#mascWords = words[1000:]
#




fem = getEncodingsForList(femWords)
masc = getEncodingsForList(mascWords) #sanity check

## so initial 0 will look like dropout
#char_embeddings.data[0] = 0 * char_embeddings.data[0]

print("Concatenating")

predictors = []
dependent = []
for vectors in plus(fem, masc):
     code = vectors[0] #torch.cat(vectors, dim=0)
    # print(code)
     predictors.append(code.data.cpu().numpy()) #.tolist())
for _ in fem:
  dependent.append(0)# if random.random() > 0.5 else 1) # sanity check
for _ in masc:
  dependent.append(1) # if random.random() > 0.5 else 1) # sanity check
     

# create logistic regression for gender

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, words_train, words_test = train_test_split(predictors, dependent, femWords+mascWords, test_size=0.1, random_state=random.randint(0,50), shuffle=True)


#quit()

from sklearn.linear_model import LogisticRegression

print(list(zip(words_train, y_train)))
print(list(zip(words_test, y_test)))
#print(x_train)
print("regression")

logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)

score = logisticRegr.score(x_test, y_test)
print(predictions)
print(y_test)
print(score)
print(set(words_train).intersection(set(words_test)))






