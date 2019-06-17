This repository contains data and code for probing the syntactic abilities 
of language models on German verb argument structures. 

The LSTM language model used here was trained on German Wikipedia by Hahn and Baroni (submitted, 2019).


## Generating test set

We generate a verb argument structure dataset using templates.



## Testing language models

1. Replication of syntactic experiment 

We replicate the gender experiment described in Hahn and Baroni (submitted, 2019).
Test sets for this experiment are found in the Stimuli folder.

```
python experiments.py --test gender
```

2. Test on verb argument structure dataset

For grammatical sentences: 
```
python experiments.py --test syntax --dataset grammatical
```

For ungrammatical sentences: 
```
python experiments.py --test syntax --dataset ungrammatical
```