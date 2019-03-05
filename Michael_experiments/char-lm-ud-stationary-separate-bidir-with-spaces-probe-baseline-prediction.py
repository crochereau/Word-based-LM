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
itos.append(" ")
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn_forward = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num, bidirectional=False).cuda()
rnn_backward = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num, bidirectional=False).cuda()


rnn_forward_drop = WeightDrop(rnn_forward, [(name, args.weight_dropout_in) for name, _ in rnn_forward.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn_forward.named_parameters() if name.startswith("weight_hh_")])
rnn_backward_drop = WeightDrop(rnn_backward, [(name, args.weight_dropout_in) for name, _ in rnn_backward.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn_backward.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn_forward, rnn_backward, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn_forward" : rnn_forward, "rnn_backward" : rnn_backward, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

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





rnn_forward_drop.train(False)
rnn_backward_drop.train(False)

#baseline_rnn_encoder_drop.train(False)


def encodeSequenceBatchForward(numeric):
      input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)

#      target_tensor_forward = Variable(torch.LongTensor(numeric).transpose(0,1)[2:].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
      embedded_forward = char_embeddings(input_tensor_forward)
      out_forward, hidden_forward = rnn_forward_drop(embedded_forward, None)
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
      
      out, hidden = (rnn_forward_drop if not backwards else rnn_backward_drop)(embedded_forward, hidden)
      out = out[-1]

 #   rnn_forward_drop.train(False)


    return output_string if not backwards else output_string[::-1]


out1, hidden1 = encodeSequenceBatchForward(encodeWord("katze"))
out2, hidden2 = encodeSequenceBatchForward(encodeWord("katzem"))
#print(torch.dot(out1[-1], out2[-1]))
#print(torch.dot(hidden1[0], hidden2[0]))
#print(torch.dot(hidden1[1], hidden2[1]))

print(torch.nn.functional.cosine_similarity(out1, out2, dim=0))
#print(torch.nn.functional.cosine_similarity(hidden1, hidden2, dim=0))
#print(torch.nn.functional.cosine_similarity(cell1, cell2, dim=0))

print("will mach")
print(predictNext(encodeSequenceBatchForward(encodeWord(" ich mach"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" du mach"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" er mach"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" sie mach"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" es mach"))))

print(keepGenerating(encodeSequenceBatchForward(encodeWord(" ich mach"))))
print(keepGenerating(encodeSequenceBatchForward(encodeWord(" du mach"))))
print(keepGenerating(encodeSequenceBatchForward(encodeWord(" er mach"))))
print(keepGenerating(encodeSequenceBatchForward(encodeWord(" sie mach"))))
print(keepGenerating(encodeSequenceBatchForward(encodeWord(" es mach"))))
print(keepGenerating(encodeSequenceBatchForward(encodeWord(" es denk"))))





print(predictNext(encodeSequenceBatchForward(encodeWord(" ich will mach"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" habe gemach"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" habe gepups"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" habe gegurk"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" habe gerief"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" ich werde pups"))))
print("Katze")
print(predictNext(encodeSequenceBatchForward(encodeWord(" eine katze"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" zwei katze"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" der kater"))))
print(predictNext(encodeSequenceBatchForward(encodeWord(" des kater"))))

print(keepGenerating(encodeSequenceBatchBackward(encodeWord(" katze ")), backwards=True))
print(keepGenerating(encodeSequenceBatchBackward(encodeWord(" kater ")), backwards=True))
print(keepGenerating(encodeSequenceBatchBackward(encodeWord(" hund ")), backwards=True))
print(keepGenerating(encodeSequenceBatchBackward(encodeWord(" neben ")), backwards=True))



quit()
                  
def padWords(words):
   maxLength = max([len(x) for x in words])
   for i, word in enumerate(words):
      if len(word) < maxLength:
         words[i] = ([0] * (maxLength - len(word))) + word #word.append(0)
   return words


# TODO train ght eRNN so that it is actually used to seeing 0's in the beginning of a sequence

def getEncodingsForList(wordsToBeEncoded):
    return getEncodingsForListGeneral(wordsToBeEncoded, encodeSequenceBatchForward)



def getEncodingsForListGeneral(wordsToBeEncoded, encodingFunction):
    modelVectors = []
    byLength = sorted(list(wordsToBeEncoded), reverse=True)

    for offset in range(0, len(wordsToBeEncoded), 100):
#      print(offset)
      codes1, codes2 = encodingFunction(padWords([encodeWord(word)[0] for word in byLength[offset:offset+100]]))
      for index, word in enumerate(byLength[offset:offset+100]):
         code1 = codes1[index].cpu()#,len(word)]
         code2 = codes2[0][0,index].cpu()#,len(word)]
         code3 = codes2[1][0,index].cpu()#,len(word)]
         modelVectors.append((code1, (code2, code3)))
    #     print((code1,code2,code3))
    return modelVectors





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

fem = getEncodingsForList(random.sample(genders["Gender=Fem"], 1000))
masc = getEncodingsForList(random.sample(genders["Gender=Masc"], 1000))

## so initial 0 will look like dropout
#char_embeddings.data[0] = 0 * char_embeddings.data[0]

print("Concatenating")

predictors = []
dependent = []
for vectors in plus(fem, masc):
     code = vectors[0] #torch.cat(vectors, dim=0)
    # print(code)
     predictors.append(code.data.cpu().numpy())
for _ in fem:
  dependent.append(0)
for _ in masc:
  dependent.append(1)
     

# create logistic regression for gender

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, dependent, test_size=0.1, random_state=0, shuffle=True)


from sklearn.linear_model import LogisticRegression

print("regression")

logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)


score = logisticRegr.score(x_test, y_test)
print(score)






