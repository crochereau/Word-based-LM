# python char-lm-ud-wiki-classify-boundaries-report-errors-upToEndOnly-lexCountBaseline.py --language english


from paths import WIKIPEDIA_HOME
from paths import LOG_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)

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


#assert args.language == "german"


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
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


#rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

#rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
#print(rnn_parameter_names)
#quit()


#rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

#output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

#char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

#modules = [rnn, output, char_embeddings]
#def parameters():
#   for module in modules:
#       for param in module.parameters():
#            yield param

#parameters_cached = [x for x in parameters()]

#optim = torch.optim.SGD(parameters(), lr=args.learning_rate, momentum=0.0) # 0.02, 0.9

#named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

#if args.load_from is not None:
#  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
#  for name, module in named_modules.items():
#      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
      numeric = [0]
      boundaries = [None for _ in range(args.sequence_length+1)]
      boundariesAll = [None for _ in range(args.sequence_length+1)]

      count = 0
      currentWord = ""
      print("Prepare chunks")
      for chunk in data:
       print(len(chunk))
       for char in chunk:
         if char == " ":
           boundaries[len(numeric)] = currentWord
           boundariesAll[len(numeric)] = currentWord

           currentWord = ""
           continue
         else:
           if boundariesAll[len(numeric)] is None:
               boundariesAll[len(numeric)] = currentWord

         count += 1
         currentWord += char
#         if count % 100000 == 0:
#             print(count/len(data))
         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
         if len(numeric) > args.sequence_length:
            yield numeric, boundaries, boundariesAll
            numeric = [0]
            boundaries = [None for _ in range(args.sequence_length+1)]
            boundariesAll = [None for _ in range(args.sequence_length+1)]





#def prepareDataset(data, train=True):
#      numeric = [0]
#      count = 0
#      for char in data:
#         if char == " ":
#           continue
#         count += 1
##         if count % 100000 == 0:
##             print(count/len(data))
#         numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
#         if len(numeric) > args.sequence_length:
#            yield numeric
#            numeric = [0]
#

# from each bath element, get one positive example OR one negative example

wordsSoFar = set()
hidden_states = []
labels = []
relevantWords = []
relevantNextWords = []
labels_sum = 0

def forward(numeric, train=True, printHere=False):
      global labels_sum
      numeric, boundaries, boundariesAll = zip(*numeric)
#      print(numeric)
 #     print(boundaries)

#      input_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[:-1].cuda(), requires_grad=False)
 #     target_tensor = Variable(torch.LongTensor(numeric).transpose(0,1)[1:].cuda(), requires_grad=False)

  #    embedded = char_embeddings(input_tensor)
   #   if train:
    #     embedded = char_dropout(embedded)

     # out, _ = rnn_drop(embedded, None)
#      if train:
#          out = dropout(out)

      for i in range(len(boundaries)): # for each batch sample
         target = (labels_sum + 10 < len(labels)/2) or (random.random() < 0.5) # decide whether to get positive or negative sample
         true = sum([((x == None) if target == False else (x not in wordsSoFar)) for x in boundaries[i][int(args.sequence_length/2):-1]]) # condidates
 #        print(target, true)
         if true == 0:
            continue
         soFar = 0
#         print(list(zip(boundaries[i], boundariesAll[i])))
         for j in range(len(boundaries[i])):
           if j < int(len(boundaries[i])/2):
               continue
           if (lambda x:((x is None if target == False else x not in wordsSoFar)))(boundaries[i][j]):
 #             print(i, target, true,soFar)
              if random.random() < 1/(true-soFar):
#                  hidden_states.append(out[j-1,i].detach().data.cpu().numpy())
                  labels.append(1 if target else 0)
                  relevantWords.append(boundariesAll[i][j])
                  relevantNextWords.append(([boundaries[i][k] for k in range(j+1, len(boundaries[i])) if boundaries[i][k] is not None]+["END_OF_SEQUENCE"])[0])
                  assert boundariesAll[i][j] is not None

                  labels_sum += labels[-1]
                  if target:
                     wordsSoFar.add(boundaries[i][j])
                  break
              soFar += 1
         assert soFar < true
#      print(hidden_states)
#      print(labels)

#      logits = output(out) 
#      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

 #     loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

      if printHere:
  #       lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(args.sequence_length, len(numeric))
   #      losses = lossTensor.data.cpu().numpy()
#         boundaries_index = [0 for _ in numeric]
         for i in range((args.sequence_length-1)-1):
 #           if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
  #             boundary = True
   #            boundaries_index[0] += 1
    #        else:
     #          boundary = False
            print((itos[numeric[0][i+1]-3], "read:", itos[numeric[0][i]-3], boundariesAll[0][i], boundariesAll[0][i+1] if i < args.sequence_length-2 else "EOS"))
         print((labels_sum, len(labels)))
     # return loss, len(numeric) * args.sequence_length



import time

devLosses = []
#for epoch in range(10000):
if True:
   training_data = corpusIteratorWiki.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   #rnn_drop.train(False)
   startTime = time.time()
   trainChars = 0
   counter = 0
   while True:
      counter += 1
      try:
         numeric = [next(training_chars) for _ in range(args.batchSize)]
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      forward(numeric, printHere=printHere, train=True)
      #backward(loss, printHere)
      if printHere:
          print((counter))
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))

      if len(labels) > 10000:
         break

predictors = hidden_states
dependent = labels


stimuli = sorted(list(zip(relevantWords, dependent)), key=lambda x:x[0])
stimuliUnique = {}
for prefix, dependent in stimuli:
    if prefix not in stimuliUnique:
       stimuliUnique[prefix] = [0,0]
    stimuliUnique[prefix][dependent] += 1

with open("/u/scr/mhahn/FAIR18/english-wiki-word-vocab.txt", "r") as inFile:
  print("reading")
  lexicon = [x.split("\t") for x in inFile.read().strip().split("\n")] #[:10000]]
print("sorting")
lexicon = sorted(lexicon, key=lambda x:x[0])
print("done")
print(lexicon[:100])

#quit()
print(stimuliUnique)

correct = 0
incorrect = 0

last = 0
predictions = []
for stimulus, isAndIsntBoundary in stimuliUnique.items():
#  print("SEARCHING FOR", stimulus)
  # find the postion of the word
  i = last
  j = len(lexicon)-1
  while True:
     mid = int((i+j)/2)
     assert mid >= i
     if i == mid:
        break
     inMid = lexicon[mid][0]
     if inMid <= stimulus:
        i = mid
     else:
        assert inMid > stimulus, (inMid, stimulus)
        j = mid
  if not lexicon[i][0].startswith(stimulus):
    predictions.append(1 if random.random() > 0.5 else 0)
    print ("NO SUFFIX FOUND", (stimulus, lexicon[i-1], lexicon[i], stimulus >= lexicon[i][0], stimulus <= lexicon[i-1][0]))
  else:
     assert not lexicon[i-1][0].startswith(stimulus), (stimulus, lexicon[i-1], lexicon[i], stimulus >= lexicon[i][0], stimulus <= lexicon[i-1][0])
     if lexicon[i+1][0] < stimulus:
           assert i+1 == len(lexicon), (lexicon[i], lexicon[i+1], stimulus, i, j)
     #assert stimulus <= lexicon[i][0] and stimulus >= lexicon[i-1][0], (stimulus, lexicon[i-1], lexicon[i], stimulus >= lexicon[i][0], stimulus <= lexicon[i-1][0])
     last = i
   
     r = i
     s = len(lexicon)-1
     while True:
        mid = int((r+s)/2)
 #       print(r, mid, s, stimulus, lexicon[r], lexicon[s])
        assert mid >= r
        if r == mid:
           break
        inMid = lexicon[mid][0]
        assert inMid >= stimulus
        if not inMid.startswith(stimulus):
           s = mid
        else:
   #        assert inMid > stimulus, (inMid, stimulus)
           r = mid
     print(stimulus, lexicon[r], lexicon[s], i, j, r, s)
     assert lexicon[s-1][0].startswith(stimulus)
     assert not lexicon[s][0].startswith(stimulus)
     # from lexicon[i] to lexicon[s-1] is the span
     if lexicon[i][0] != stimulus:
         predictions.append(1)
     else:
         countsWord = int(lexicon[i][1])
         countsOther = sum([int(lexicon[q][1]) for q in range(i+1, s)])
         prediction = 1 if (countsWord > countsOther) else 0
         print(stimulus, countsWord, countsOther, isAndIsntBoundary)
         if prediction == 0:
           correct += isAndIsntBoundary[0]
           incorrect += isAndIsntBoundary[1]
         elif prediction == 1:
           correct += isAndIsntBoundary[1]
           incorrect += isAndIsntBoundary[0]
         print(correct, incorrect, correct/(1+correct+incorrect))

#         predictions.append()
         
#     quit()
#assert len(predictions) == len(dependent)
#print(len(predictions))
#print(sum([x==y for x, y in zip(predictions, dependent)])/len(predictions))
quit()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, words_train, words_test, next_words_train, next_words_test = train_test_split(predictors, dependent, relevantWords, relevantNextWords, test_size=0.91, random_state=0, shuffle=True)


from sklearn.linear_model import LogisticRegression

print("regression")

logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)


score = logisticRegr.score(x_test, y_test)

errors = []
for i in range(len(predictions)):
    if predictions[i] != y_test[i]:
          errors.append((y_test[i], (words_test[i], next_words_test[i], predictions[i], y_test[i])))
for error in errors:
   if error[0] == 0:
      print(error[1][0]+"|"+error[1][1])
for error in errors:
   if error[0] == 1:
      print(error[1][0]+" "+error[1][1])



print(len(predictions))

print("Balance ",sum(y_test)/len(y_test))
print(score)



