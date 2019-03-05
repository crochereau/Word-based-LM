from paths import WIKIPEDIA_HOME

bigrams = {}

lastLine = "."
counter = 0
with open(WIKIPEDIA_HOME+"german-train-tagged.txt", "r") as inFile:
    for line in inFile:
      if line.startswith("<"): # == "<br>\n" or line == "<br >\n":
         lastLine = "."
         continue
      counter += 1
      if counter % 100000 == 0:
         print(counter/819597764)
#      assert "\t" in line, [x for x in line]
      line = line[:line.index("\t")].lower()
      if lastLine not in bigrams:
         bigrams[lastLine] = {"_TOTAL_" : 0}
      bigrams[lastLine][line] = bigrams[lastLine].get(line, 0) + 1
      bigrams[lastLine]["_TOTAL_"] = bigrams[lastLine].get("_TOTAL_", 0) + 1
      lastLine = line
#      if counter > 1000:
 #          break
with open("/checkpoint/mhahn/bigrams-german.txt", "w") as outFile:
   for left in bigrams:
       print(f"#START#{left}", file=outFile)
       for entry, value in bigrams[left].items():
            if value == 1:
                 continue
            print(f"{entry}\t{value}", file=outFile)



