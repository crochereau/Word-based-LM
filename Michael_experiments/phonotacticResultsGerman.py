


pairs = ["bl,bd", "bu,bt", "do,dd", "fl,fd", "fr,fn", "fu,ft", "kl,kt", "pl,pt", "po,pt", "tu,tt", "vo,vd", "zu,zt"]
pairs = sorted(pairs)

import sys
import subprocess
for pair in pairs:
  process = subprocess.Popen("python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-two-sequences-start.py --language german --hidden_dim 1024 --layer_num 2 --learning_rate 2.0 --iterations 1epoch".split(" ") + ["--sequences", pair, "--printing", "False"], stdout=sys.stdout)
  process.wait()
