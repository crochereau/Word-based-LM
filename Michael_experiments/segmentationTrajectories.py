from paths import MODELS_HOME

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)

args=parser.parse_args()
print(args)

i = 0
import os.path
import subprocess

while os.path.isfile(MODELS_HOME+"/"+args.load_from+"_EPOCH_"+str(i)+".pth.tar"):
   subprocess.call(["python", "lm-acqdiv-segmentation-analyses-morph-pos.py", "--language", args.language, "--load-from", args.load_from+"_EPOCH_"+str(i)])
   i += 1


