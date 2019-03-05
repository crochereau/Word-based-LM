
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
import random


args=parser.parse_args()
print(args)





from acqdivReader import AcqdivReader, AcqdivReaderPartition

acqdivCorpusReader = AcqdivReader(args.language)


import syllabificationJapanese

syllables = {}	

for chunk in acqdivCorpusReader.iterator():
#       print(len(chunk))

       # tokenize chunk into valid syllables
       words = chunk.split(" ")
       for word in words:
          if word == "\n":
              continue  
          if len(word) == 0:
              continue
          if word == "n":
             continue
          if word == "???":
             continue
          syllabification = syllabificationJapanese.syllabify(word)
          if syllabification is None:
              continue
          for syll in syllabification:
             if syll[0] is None:
                 continue
             syll = syll[0]+syll[1]+syll[2]
             syllables[syll] = syllables.get(syll, 0) + 1

print(syllables)
