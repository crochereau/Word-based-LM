
# coding: utf-8

# In[311]:

from paths import WIKIPEDIA_HOME
from paths import MODELS_HOME

import argparse
import corpusIteratorWikiWords
import random
import torch
from weight_drop import WeightDrop
import numpy as np
from corpusIterator import CorpusIterator
import math
print(torch.__version__)


# In[312]:

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


# In[313]:

char_vocab_path = "vocabularies/german-wiki-word-vocab-50000.txt"


# In[314]:

""" Generate a vocabulary dictionary """

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


# In[316]:

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[317]:

rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).to(device)

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)

rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])
output = torch.nn.Linear(args.hidden_dim, len(itos)+3).to(device)
char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).to(device)
logsoftmax = torch.nn.LogSoftmax(dim=2)
#softmax = torch.nn.Softmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(reduction='none', ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]


# In[318]:

def parameters():
    for module in modules:
        for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]


# In[319]:

learning_rate = args.learning_rate
optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9
named_modules = {"rnn": rnn, "output": output, "char_embeddings": char_embeddings, "optim": optim}


# In[320]:

print("Loading model")
if args.load_from is not None:
# Loads model from checkpoint
#  Forcibly loads model onto cpu. Delete map_location argument to have storage on gpu.
    checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar", map_location=device)
    for name, module in named_modules.items():
        print(checkpoint[name].keys())
        module.load_state_dict(checkpoint[name])
else:
    assert False


# In[321]:

rnn_drop.train(False)
lossModule = torch.nn.NLLLoss(reduction='none', ignore_index=0)


# ### Test implementation

# In[322]:

def load_sentences(input_path):
    """ 
    Input: str, path to input file
    Output: loaded file 
    """
    with open(input_path, "r") as f:
        output = f.read()
            
    return output


# In[246]:

def tokenizer(sentences):
    
    """
    Input: loaded text file
    Output: tokenized text
    """
    sentences = sentences.replace(".", ". <eos>") # adding end of sentence symbols for tokenization
    #print(sentences)
    
    sentences = sentences.split("<eos>") # separates sentences
    sentences.pop() # removes last element (a whitespace)
    print("number of test sentences: ", len(sentences))
    
    for i in range(len(sentences)): # tokenizes sentences
        sentences[i] = sentences[i].replace(",", " ,").replace(".", " .").replace(":", " :").replace("?", " ?").replace("!", " !").replace(";", " ;")
        sentences[i] = sentences[i].lower()
        sentences[i] = sentences[i].split()
    print(sentences[:10])
    
    tokenized_sentences = [[] for i in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            tokenized_sentences[i].append(stoi[word]+3 if word in stoi else 2)
    
    return tokenized_sentences


# In[323]:

def compute_logprob(numeric):
    """  
    Compute log probabilities of words of input sentences.
    Input: list of lists encoding sentences 
    Outputs: 
    - vector containing padded sentences, 
    - prediction table containing log probabilities for each word of vocabulary, for each word of input sentences
        size: length_of_longer_input_sentence * number_of_sentences * vocabulary_size 
    """
    
    maxLength = max([len(x) for x in numeric])
    print("maxLength:", maxLength)
    
    # padding shorter sentences with zeros 
    for i in range(len(numeric)):
        while len(numeric[i]) < maxLength:
            numeric[i].append(0)
    print("numeric:", numeric)
            
    input_tensor_forward = torch.tensor([[0]+x for x in numeric], dtype = torch.long, device=device, requires_grad=False).transpose(0,1)
     
    target = input_tensor_forward[1:]
    input_cut = input_tensor_forward[:-1]
    embedded_forward = char_embeddings(input_cut)
    out_forward, hidden_forward = rnn_drop(embedded_forward, None)
    prediction = logsoftmax(output(out_forward))
    predictions = prediction.detach().numpy()
    
    losses = lossModule(prediction.reshape(-1, len(itos)+3), target.reshape(-1)).reshape(maxLength, len(numeric))
    losses = losses.sum(0).data.cpu().numpy()
    
    return numeric, predictions, losses


# In[248]:

def per_sentence_probabilities(padded_sentences, log_predictions):
    """ 
    Get per-word log probabilities for each input sentence from prediction table 
    and computes per-sentence probabilities.
    Input: 
    Output: 
    """
    
    word_log_probs = [[] for i in range(len(padded_sentences))]

    # get per-word log probabilities from log_predictions table
    for i in range(len(padded_sentences)):
        for j in range(len(padded_sentences[i])):
            k = padded_sentences[i][j]
            if k != 0:      # because of padding with zeros
                word_log_probs[i].append(log_predictions[j][i][k])
    
    """
    Printing intermediatexs results
    
    print("%s sentences" %len(padded_sentences))
    for i in range(len(padded_sentences)):
        print("number of tokens in sentence %s:" %(i+1), len(word_log_probs[i]))
    
    print("per-word log probabilities: ")
    for i in range(len(padded_sentences)):
        print("sentence %s " %(i+1), word_log_probs[i])
    """
    
    sentence_probs = []
    for i in range(len(word_log_probs)):
        sentence_probs.append(math.exp(sum(word_log_probs[i])))

    return sentence_probs


"""
# In[196]:

sentences_path = "input_sentences/sentences.txt"
test_sentences = load_sentences(sentences_path)


# In[197]:

tokens = tokenizer(test_sentences)
print(tokens)
len(tokens[0])


# In[199]:

numericalized_sentences, logprobs, loglosses = compute_logprob(tokens)
print(len(numericalized_sentences))
#print(logprobs.size()) # size: length of longest sentence, number of sentences, vocabulary size


# In[200]:

print(per_sentence_probabilities(numericalized_sentences, logprobs))

"""


# ### Test with gender stimuli

# In[324]:

gender_path = "stimuli/german-gender-Gender=Neut-sehr + extrem + adjective-noOOVs.txt"
gender_test_sentences = load_sentences(gender_path)                                                                                                     


# In[325]:

def gender_tokenizer(sentences):
    
    """
    Input: loaded text file
    Output: tokenized text
    Difference with tokenizer(): gender stimuli have no punctuation -> slightly different preprocessing
    """
    #sentences = sentences.replace(".", ". <eos>") # adding end of sentence symbols for tokenization
    #print(sentences)
    
    tokens = sentences.split()
    #print("number of test sentences: ", len(sentences))
    tokenized_sentences = [tokens[x:x+5] for x in range(0, len(tokens), 5)]
    #print(tokenized_sentences)
   
    
    encoded_sentences = [[] for i in range(len(tokenized_sentences))]
    for i in range(len(tokenized_sentences)):
        for j in range(len(tokenized_sentences[i])):
            word = tokenized_sentences[i][j]
            encoded_sentences[i].append(stoi[word]+3 if word in stoi else 2)
    
    return encoded_sentences


# In[326]:

gender_tokens = gender_tokenizer(gender_test_sentences)

# Adding a dot at the begining and the end of each stimulus encoded in gender_tokens
# To match Michael's preparation of stimuli
# Stimuli now have this shape: . {stimulus} .

for i in range(len(gender_tokens)):
    gender_tokens[i].insert(0,3)
    gender_tokens[i].append(3)

print(gender_tokens)


# In[327]:

print(len(gender_tokens))


# In[328]:

# We check if tokenization went wrong in the stimuli dataset
for idx, value in enumerate(gender_tokens):
    if len(value) != 7:
        print(idx, len(value))
            
# No output: tokenization went fine


# In[329]:

numericalized_gender_sentences, gender_logprobs, gender_losses = compute_logprob(gender_tokens)


# Many words are encoded with 2: OOV. 
# 2 might get a high probability -> bias results? 

# In[330]:

print("number of sentences:", len(numericalized_gender_sentences))
print("shape of log probabilities prediction:", gender_logprobs.shape) 
print("size of losses: ", len(gender_losses))
#print(gender_logprobs.size())


# In[266]:

"""
print(len(gender_results)/3)
der_loss_list = gender_losses[0::3]
die_loss_list = gender_losses[1::3]
das_loss_list = gender_losses[2::3]
print(len(der_loss_list), len(die_loss_list), len(das_loss_list))

der_loss = sum(der_loss_list)  #/sum(gender_results)
die_loss = sum(die_loss_list)  #/sum(gender_results)
das_loss = sum(das_loss_list)  #/sum(gender_results)
print("der probability:", der_loss)
print("die probability:", die_loss)
print("das probability:", das_loss)
"""


# In[331]:

gender_results = per_sentence_probabilities(numericalized_gender_sentences, gender_logprobs)
print(gender_results)


# In[332]:

#for idx, value in enumerate(gender_results):
    #print(idx, value)
    #if value == 1.0:
        #del gender_results[idx]


# In[333]:

# Probability of each gender

print(len(gender_results)/3)
der_list = gender_results[0::3]
die_list = gender_results[1::3]
das_list = gender_results[2::3]
print(len(der_list), len(die_list), len(das_list))

der_prob = sum(der_list)/sum(gender_results)
die_prob = sum(die_list)/sum(gender_results)
das_prob = sum(das_list)/sum(gender_results)
print("der probability:", der_prob)
print("die probability:", die_prob)
print("das probability:", das_prob)


# In[310]:

print(sum(gender_results))
print(sum(die_list))
print(sum(das_list))
print(sum(der_list))

"""
# ### Checking what the max log probabilities are and to which words they correspond

# In[70]:

# to finish
max_probs_values, max_probs_idx = torch.max(log_prediction, 2)
print(max_probs_values.size(), max_probs_idx.size())
print(max_probs_values, max_probs_idx)

"""

# In[ ]:

"""
def compute_logprob(numeric):

    maxLength = max([len(x) for x in numeric])
    print("maxLength:", maxLength)
    
    # padding shorter sentences with zeros 
    for i in range(len(numeric)):
        while len(numeric[i]) < maxLength:
            numeric[i].append(0)
    print("numeric:", numeric)
            
    input_tensor_forward = torch.tensor([[0]+x for x in numeric], dtype = torch.long, device=device, requires_grad=False).transpose(0,1)
     
    target = input_tensor_forward[1:]
    input_cut = input_tensor_forward[:-1]
    embedded_forward = char_embeddings(input_cut)
    out_forward, hidden_forward = rnn_drop(embedded_forward, None)
    prediction = logsoftmax(output(out_forward))
    #values_best_log_prob, indices_best_log_prob = torch.max(prediction,2)
    #print(values_best_log_prob, indices_best_log_prob)
    #print(values_best_log_prob.size(), indices_best_log_prob.size())
    #losses = lossModule(prediction.reshape(-1, len(itos)+3), target.reshape(-1)).reshape(maxLength, len(numeric))
    #losses = losses.sum(0).data.cpu().numpy()
    return numeric, prediction # values_best_log_prob
"""

