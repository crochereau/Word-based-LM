from paths import WIKIPEDIA_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME


# Clear evidence that the model isn't leveraging evidence about the subcategorization of the verb.

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
def doChoiceList(xs, printHere=True):
    if printHere:
      for x in xs:
         print(x)
    losses = choiceList([encodeWord(x) for x in xs]) #, encodeWord(y))
    if printHere:
      print(losses)
    return np.argmin(losses)


def doChoice(x, y):
    print(x)
    print(y)
    losses = choice(encodeWord(x), encodeWord(y))
    print(losses)
    return 0 if losses[0] < losses[1] else 1
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

correctFirst = [0,0]
correctSecond = [0,0]
correctThird = [0,0]
correctThirdSingPlur = [0,0]
correctThirdSingPlurFar = [0,0]
correctThirdPlurSingFar = [0,0]



with open("germanConjugationCleaned.txt", "r") as inFile:
    conj = inFile.read().strip().replace("\xa0", " ").split("###")
    for verb in conj:
        verb = verb.split("%")
        if len(verb) == 1:
             print(verb)
             continue
        lemma = verb[0].strip()
        praesens = [x for x in verb[1].strip().split("\n")]
        praesens = [[[z.strip().split(" ") for z in  y.split(", ")] for y in x.split("\t")[1:]] for x in praesens]
        praeteritum = [[[z.strip().split(" ") for z in  y.split(", ")] for y in x.split("\t")[1:]] for x in verb[2].strip().split("\n")]
        withParticiple = verb[3].strip()
        # first person present
        presentForms =  set()
        for x in praesens:
           assert presentForms is not None
           for j in x:
            assert presentForms is not None
            for i in j:
              if len(i) >=2: # includes particle verbs
                 
                 form = "".join(i[1:])
                 assert " " not in form, praesens
                 if i[0] not in ["ich", "du", "er/sie/es", "wir", "ihr", "sie"]:
                   if "veraltet:" not in i:
                      print(i)
                   continue
                 if "veraltet:" in i:
                    continue
                 i = [z for z in i if len(z) > 0 and z != "[1]"]
                 
                 if len(i) not in [2,3] and i[2] != "wieder":
                   print(["ERROR", i])
                   continue
                 presentForms.add(form)

#            if presentForms == None:
#               break 
#           if presentForms == None:
#                break
#        if presentForms == None:
#           continue
#        continue
        if len(presentForms) > 0:
            if len(praesens[0][0][0]) < 2: # no first person form 
               continue
            firstSingularForm = "".join(praesens[0][0][0][1:])
            stimuli = [".ich"+form+"." for form in [firstSingularForm] + list(presentForms-set(firstSingularForm))]
            result = doChoiceList(stimuli, printHere=True)
            correctFirst[0] += (1 if result  == 0 else 0)
            correctFirst[1]+=1
            print(correctFirst[0]/correctFirst[1])
         
            if len(praesens[1][0][0]) < 2: # no second person form 
               continue
            secondSingularForm = "".join(praesens[1][0][0][1:])
            stimuli = [".du"+form+"." for form in [secondSingularForm] + list(presentForms-set(secondSingularForm))]
            result = doChoiceList(stimuli, printHere=True)
            correctSecond[0] += (1 if result  == 0 else 0)
            correctSecond[1]+=1
            print(correctSecond[0]/correctSecond[1])
 
            if len(praesens[2][0][0]) < 2: # no third person form 
               continue
            thirdSingularForm = "".join(praesens[2][0][0][1:])
            stimuli = [".er"+form+"." for form in [thirdSingularForm] + list(presentForms-set(thirdSingularForm))]
            result = doChoiceList(stimuli, printHere=True)
            correctThird[0] += (1 if result  == 0 else 0)
            correctThird[1]+=1
            print(correctThird[0]/correctThird[1])



            # third singular vs plural
            thirdSingularForm = "".join(praesens[2][0][0][1:])
            thirdPluralForm = "".join(praesens[5][0][0][1:])
            stimuli = [f'.er{thirdSingularForm}.', f'.er{thirdPluralForm}.']
            result = doChoiceList(stimuli, printHere=True)
            correctThirdSingPlur[0] += (1 if result  == 0 else 0)
            correctThirdSingPlur[1]+=1
            print(correctThirdSingPlur[0]/correctThirdSingPlur[1])

 
            # third singular vs plural
            thirdSingularForm = "".join(praesens[2][0][0][1:][::-1])
            thirdPluralForm = "".join(praesens[5][0][0][1:][::-1])
            subject = "er"
            intervening = "jaextremunglaublichgerne" #"gerne"
            matrix = "ichsage,dass"
            post = "" #,obwohldasfalschist"
            stimuli = [f'.{matrix}{subject}{intervening}{thirdSingularForm}{post}.', f'.{matrix}{subject}{intervening}{thirdPluralForm}{post}.']
            result = doChoiceList(stimuli, printHere=True)
            correctThirdSingPlurFar[0] += (1 if result  == 0 else 0)
            correctThirdSingPlurFar[1]+=1
            print("ThirdSingPlurFor", correctThirdSingPlurFar[0]/correctThirdSingPlurFar[1])

#             # third singular vs plural
#            thirdSingularForm = "".join(praesens[2][0][0][1:][::-1])
#            thirdPluralForm = "".join(praesens[5][0][0][1:][::-1])
#            subject = "dieleute"
#            intervening = "gerne" #"gerne"
#            matrix = "ersagt,dass"
#            post = "" #,obwohldasfalschist"
#            stimuli = [f'.{matrix}{subject}{intervening}{thirdSingularForm}{post}.', f'.{matrix}{subject}{intervening}{thirdPluralForm}{post}.']
#            result = doChoiceList(stimuli, printHere=True)
#            correctThirdPlurSingFar[0] += (1 if result  == 1 else 0)
#            correctThirdPlurSingFar[1]+=1
#            print("ThirdPlurSingFor", correctThirdPlurSingFar[0]/correctThirdPlurSingFar[1])

                     

