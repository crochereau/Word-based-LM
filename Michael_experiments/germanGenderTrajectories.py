
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)

args=parser.parse_args()
print(args)

i = 0
import os.path
import subprocess




while os.path.isfile("/checkpoint/mhahn/wiki-german-nospaces-bugfix-checkpoints_CHECKPOINT"+str(i)+".pth.tar"):
   subprocess.call(["python", "char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki.py", "--language", "german", "--batchSize", "128", "--char_embedding_size", "100", "--hidden_dim", "1024", "--layer_num", "2", "--weight_dropout_in", "0.1", "--weight_dropout_hidden", "0.35", "--char_dropout_prob", "0.0", "--char_noise_prob", "0.01", "--learning_rate", "0.2", "--load-from", "wiki-german-nospaces-bugfix-checkpoints_CHECKPOINT"+str(i)])
   i += 1

