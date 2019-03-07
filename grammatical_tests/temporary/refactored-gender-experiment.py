"""Gender experiment - WordNLM"""

from paths import WIKIPEDIA_HOME
from paths import MODELS_HOME

import argparse
import random
import torch
from weight_drop import WeightDrop
import numpy as np
print(torch.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="german")
parser.add_argument("--load-from", dest="load_from", type=str, default="wiki-german-nospaces-bptt-words-966024846")
#parser.add_argument("--load-from-baseline", dest="load_from_baseline", type=str)
#parser.add_argument("--save-to", dest="save_to", type=str)
# parser.add_argument("--batchSize", type=int, default=random.choice([128, 128, 256]))
parser.add_argument("--batchSize", type=int, default=random.choice([128]))
# parser.add_argument("--char_embedding_size", type=int, default=random.choice([100, 200, 300]))
parser.add_argument("--char_embedding_size", type=int, default=random.choice([200]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
# parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0, 0.0, 0.0, 0.01, 0.05, 0.1]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.1]))
# parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0, 0.05, 0.15, 0.2]))
parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.2]))
# parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0, 0.0, 0.001, 0.01, 0.01]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0]))
# parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.0]))
parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.01]))
# parser.add_argument("--learning_rate", type = float, default= random.choice([0.8, 0.9, 1.0,1.0,  1.1, 1.1, 1.2, 1.2, 1.2, 1.2, 1.3, 1.3, 1.4, 1.5]))
parser.add_argument("--learning_rate", type=float, default=random.choice([0.2]))
parser.add_argument("--myID", type=int, default=random.randint(0, 1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50, 50, 80]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([0.5, 0.7, 0.9, 0.95, 0.98, 0.98, 1.0]))

args=parser.parse_args([])
print(args)


assert "word" in args.load_from, args.load_from

char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt"}[args.language]


""" Words anf characters of vocabulary are mapped to indices.
itos: integer -> string
stoi: string -> integer
Stoi will be used to encode words and characters into numbers """

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])

print(itos[3-3])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).to(device)
rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)


rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])
output = torch.nn.Linear(args.hidden_dim, len(itos)+3).to(device)
# nn.Embedding takes 2 arguments: vocabulary size, dimensionality of embedding
# vocabulary size = len(itos) = 50.000
# dimensionality of embedding = args.char_embedding_size = 200
char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).to(device)
logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(reduction='none', ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]

def parameters():
    for module in modules:
        for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]


learning_rate = args.learning_rate
optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9
named_modules = {"rnn": rnn, "output": output, "char_embeddings": char_embeddings, "optim": optim}

print("Loading model")
if args.load_from is not None:
#""" Loads model from checkpoint """
    checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar", map_location=device)
    for name, module in named_modules.items():
        print(checkpoint[name].keys())
        module.load_state_dict(checkpoint[name])
else:
    assert False

# evaluate model
rnn_drop.train(False)

# computes loss
lossModule = torch.nn.NLLLoss(reduction='none', ignore_index=0)



################################


def encodeWord(word):  
    """Encodes each letter of a word into a number using stoi dictionary"""
    numeric = [[]]
    for char in word:
        numeric[-1].append(stoi[char]+3 if char in stoi else 2) # 1 is for OOV words
    return numeric


def choiceList(numeric):
    
    #""" Computes sentence probabilities 
    #    Input: list of lists encoding sentences """
    
    for x in numeric: 
        assert len(x) == 1
    numeric = [x[0] for x in numeric]
    print(numeric)
    maxLength = max([len(x) for x in numeric])
    print("maxLength:", maxLength)
    
    # padding shorter sentences with zeros 
    for i in range(len(numeric)):
        while len(numeric[i]) < maxLength:
            numeric[i].append(0)
    input_tensor_forward = torch.tensor([[0]+x for x in numeric], dtype = torch.long, device=device, requires_grad=False).transpose(0,1) 
    target = input_tensor_forward[1:]
    input_cut = input_tensor_forward[:-1]

    #Embedding of encoded words
    embedded_forward = char_embeddings(input_cut)
    out_forward, hidden_forward = rnn_drop(embedded_forward, None)

    #Computes log probabilities of sequences
    prediction = logsoftmax(output(out_forward))
    print(prediction.size())

    #Computes losses
    #values_best_log_prob, indices_best_log_prob = torch.max(prediction,2)
    losses = lossModule(prediction.reshape(-1, len(itos)+3), target.reshape(-1)).reshape(maxLength, len(numeric))
    print(losses.size())
    losses = losses.sum(0).data.cpu().numpy()
    print(losses.shape)

    return losses

#sentences_list = [sentence1, sentence2]
#per_word_losses, per_word_logprobs = choiceList(sentences_list)
#per_sentence_logprobs = logprobs.sum(0)
#per_sentence_probs = torch.exp(per_sentence_logprobs)
    
def doChoiceList(xs, printHere=True):
    if printHere:
        for x in xs:
            print(x)
    # I add prediction
    losses = choiceList([encodeWord(x.split(" ")) for x in xs]) #, encodeWord(y))
    if printHere:
        print(losses)
    return np.argmin(losses)


# pour la fin
def doChoiceListLosses(xs, printHere=True):
    if printHere:
        for x in xs:
            print(x)
    losses = choiceList([encodeWord(x.split(" ")) for x in xs]) #, encodeWord(y))
    if printHere:
        print(losses)
    return losses


def genderTest(mode):

    results = [[0,0,0] for _ in range(3)]
    for genderIndex, gender in enumerate(["Gender="+x for x in ["Masc", "Fem", "Neut"]]):
        
        with open(f"stimuli/german-gender-{gender}-{mode}-noOOVs.txt", "r") as inFile:
            counter = 0
            while True:
                counter += 1
                try:
                    stimulusDer = next(inFile).strip()
                except StopIteration:
                    break
                stimulusDie = next(inFile).strip()
                stimulusDas = next(inFile).strip()
                results[genderIndex][doChoiceList([f". {stimulusDer} .", f". {stimulusDie} .", f". {stimulusDas} ."], printHere=(random.random() > 0.98))] += 1
                if random.random() > 0.98:
                    print([[round(x/(counter if genderIndex == i else 1), 2) for x in results[i]] for i in range(len(results))])
            results[genderIndex] = [x/counter for x in results[genderIndex]]

    return results


confusion1 = genderTest("nothing")
#confusion2 = genderTest("adjective")
#confusion3 = genderTest("sehr + adjective")
#confusion4 = genderTest("sehr + extrem + adjective")

print(confusion1)
#print(confusion2)
#print(confusion3)
#print(confusion4)
                                              
                                              


                                             


# Command for running the Gender experiment
# python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-WORDS.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846
