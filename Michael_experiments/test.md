

# Clustering Character Embeddings


# Phonotactics
Training:
```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-two-sequences-1epoch.py --language german  --save-to wiki-german-nospaces-bptt-SEQUENCE-ITERATIONS --hidden_dim 1024 --layer_num 2 --learning_rate 2.0 --sequences bu,bt
```

```
Testing:
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-two-sequences-start.py --language german --hidden_dim 1024 --layer_num 2 --learning_rate 2.0 --sequences bu,bt --iterations 1epoch
```




# Word boundary classification

```
python char-lm-ud-wiki-classify-boundaries.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412
```

```
python char-lm-ud-wiki-classify-boundaries-baseline.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412
```



