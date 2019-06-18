This repository contains the test set and code for probing the syntactic abilities 
of language models on German verb argument structures. 

The LSTM language model used here was trained on German Wikipedia by Hahn and Baroni (submitted, 2019).


## Generating test set

The verb argument structure dataset is generated using template sentences.

```
python generate_datasets.py
```


## Training language models

**1. LSTM**

```
python train.py -language german --batch_size 128 --word_embedding_size 200 --hidden_dim 1024 --layer_num 2 
--weight_dropout_in 0.001 --weight_dropout_hidden 0.15 --char_dropout_prob 0.1 --char_noise_prob 0.01 --learning_rate 0.9
```

**2. Unigram model**

```
awk '{n[$1]++;N++}END{for(w in n)print w,n[w],n[w]/N}' german-train-tagged.txt > unigrams.txt
```


**3. Bigram model**

```
awk 'BEGIN{prev=".";}{bigram[tolower(prev)" "tolower($1)]++;unigram[tolower(prev)]++; prev=tolower($1);l++}
END{for (b in bigram) {split(b,a," ");u=unigram[a[1]];print a[1],a[2],u+0,bigram[b]+0,bigram[b]/unigram[u]}}' 
german-train-tagged.txt > bigrams.txt
```


## Testing language models

**1. Replication of syntactic experiment**

We replicate the gender experiment described in Hahn and Baroni (submitted, 2019).
Test sets for this experiment are found in the Stimuli folder.

```
python experiments.py --test gender
```

**2. Test on verb argument structure dataset**

For grammatical sentences: 
```
python experiments.py --test syntax --dataset grammatical
```

For ungrammatical sentences: 
```
python experiments.py --test syntax --dataset ungrammatical
```