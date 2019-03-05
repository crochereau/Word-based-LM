# python phonotacticResultsItalian.py  | grep ">>>" > phono-italian-results.txt


pairs = ["br,bt", "bu,bd", "dr,dt", "du,dt", "fr,ft", "fu,ft", "pr,pt", "pu,pt", "tu,td", "vu,vd", "zu,zt"]
import sys
import subprocess
for pair in pairs:
  process = subprocess.Popen("python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-two-sequences-start.py --language italian --hidden_dim 1024 --layer_num 2 --learning_rate 2.0 --iterations 5epochs".split(" ") + ["--sequences", pair, "--printing", "False"], stdout=sys.stdout)
  process.wait()
