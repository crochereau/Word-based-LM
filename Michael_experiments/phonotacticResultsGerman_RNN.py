


pairs = ["bl,bd", "bu,bt", "do,dd", "fl,fd", "fr,fn", "fu,ft", "kl,kt", "pl,pt", "po,pt", "tu,tt", "zu,zt"]
pairs = sorted(pairs)

import sys
import subprocess
for pair in pairs:
  process = subprocess.Popen("python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-two-sequences-start-rnn.py --language german --hidden_dim 2048 --layer_num 2 --learning_rate 2.0 --char_embedding_size 50 --iterations 1epoch".split(" ") + ["--sequences", pair, "--printing", "False"], stdout=sys.stdout)
  process.wait()
