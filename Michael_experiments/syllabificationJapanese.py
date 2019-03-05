# Heuristic syllabifier for Japanese


vowels = ['a','e','i','o','u', "ɘ"]
validCodas = ['n', "q", "m", "ŋ"]

# phonotactic rule: mb reflects Nb
validInitials = ["tch", "ʃ", "tʃ","ts", "zh", "ch", "sh", "ky", "gy", "ry", "my","ny", "hy", "by", "py", "ty", "k", "t"]

def validSyllable(x):
   if x in ["^", "_"]:
       return (None, None, None, x)
   initial = []
   nucleus = []
   coda = []
   i = 0
   while i < len(x) and x[i] not in vowels:
      initial.append(x[i])
      i += 1
   while i < len(x) and (x[i] in vowels or x[i] == ":"):
      nucleus.append(x[i])
      i += 1
#   print((x, initial, nucleus))
   if len(set(initial)) > 1 and "".join(initial) not in validInitials and not (initial[0] == initial[1] and "".join(initial[1:]) in validInitials) and not (initial[0] == "d" and initial[1] == "t") and not (("".join(initial)).replace(":", "") in validInitials):
       return None 
   if len(set(nucleus)) > 1 and not (len(nucleus) == 2 and nucleus[1] == ":"):
       return None
   if len(nucleus) == 0:
       return None
   if i < len(x) and i+1 == len(x) and x[i] in validCodas:
      coda = x[i]
      return ("".join(initial), "".join(nucleus), x[i])
   elif i == len(x):
     return ("".join(initial), "".join(nucleus), "")
   else:
     return None
   
    
def syllabify(word):
          word = word.replace("^", "")
          if word.endswith("t") or word.endswith("k"):
             return None # hopeless
          if word in validCodas:
             return None
          lastEnd = len(word)
          i = len(word)-1
          syllabification = []
          while i >= 0:
             while i >= 0 and not validSyllable(word[i:lastEnd]):
  #             print(("BAD", word[i:lastEnd]))
               i -= 1
             while i >= 0 and validSyllable(word[i:lastEnd]):
 #              print(("GOOD", word[i:lastEnd]))
               i -= 1
             result = (word[i+1:lastEnd])
             syllabification.append(validSyllable(result))
             #print((result, validSyllable(result)))
             lastEnd = i+1
           
          result = (word[:lastEnd])
          if syllabification[-1] is None:
              print(".................\n"+word)
#              print(syllabification[::-1])
 #             print(result)
              return None
          else:
              return syllabification[::-1]        
         
