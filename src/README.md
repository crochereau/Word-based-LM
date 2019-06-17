## Train Language Model

These are the commands used for the random hyperparameter search. Parameters other than those specified will be randomized.

WNLM:

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words.py --language german  --save-to wiki-german-nospaces-bptt-rnn-MYID --hidden_dim 1024 --layer_num 2 
```

## Morphosyntax

### Gender

```
WNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-WORDS.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846
```

### Case agreement

```
WNLM:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-WORDS.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846
```






