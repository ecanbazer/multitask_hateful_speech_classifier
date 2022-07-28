import pandas as pd
from tqdm import tqdm
from redditscore import tokenizer
from string import punctuation

def isNaN(string):
    return string != string

def remove_some_punc(s):
    puncs = punctuation.replace(',', '').replace('.','').replace("'", "").replace('"', '').replace('!','').replace('?', '')
    for char in s:
        if char in puncs:
            s = s.replace(char, '')
    return s

#create hard-matched otg data
def get_hate_base_lexs(file_hb):

    hb_lex = pd.read_csv(file_hb,sep="\t",header=0)

    hate_term = hb_lex[["term","plural_of"]]

    hate_terms = []
    phrase_terms = []     

    hate_terms.extend(hate_term.iloc[:,0].values)   

    plur_term = hate_term.iloc[:,1].values

    plur_term = [pl for pl in plur_term if not isNaN(pl)]

    hate_terms.extend(plur_term)

    hate_terms = [hte.lower() for hte in hate_terms if 'gay' not in hte.lower()]

    return hate_terms


def prepare_otg_data(inp_data, label_file, lexicon_file, out_file):
   crazy_tokenizer = tokenizer.CrazyTokenizer(remove_punct = False,normalize=2,lowercase=True,decontract=True,urls='',hashtags='split',remove_breaks=True,latin_chars_fix=True,subreddits='')

   
   data_l = list(pd.read_csv(inp_data, sep = '\t', header=None, lineterminator='\n')[0])

   data = []
   for i in data_l:
    data.append(remove_some_punc(' '.join(crazy_tokenizer.tokenize(i))))
   

   labels = list(pd.read_csv(label_file, sep = '\t', header=None, lineterminator='\n')[0])

   data_hate = []
   data_non_hate = []
   for ind in range(len(labels)):
     if labels[ind] == 1:
       data_hate.append(data[ind])  #Only hateful instances used
     else:
       data_non_hate.append(data[ind])  




   hate_terms = get_hate_base_lexs(lexicon_file)
   hate_terms = list(set(hate_terms))
#   written_wrds = []
   ind_sen = 0
   with open('temp1.txt','w') as f:
     for dat in data_non_hate:
         for word in dat.split(" "):
             f.write(word + " " + "O" + "\n")
         f.write('\n')

   with open(out_file,'w') as f:
     for dat in data_hate:  
       ind_sen+=1
       tar_ind = []
       temp_str = "N"*len(dat) #This string keeps track of the positions that should be tagged as OTG
       for h in hate_terms:
           if h in dat:
               st_all = []
               for i in range(len(dat)):
                   if dat.startswith(h,i): #To obtain all positions in the sentence where the hate term occurs from start of a word and ends at the end position of a word or at the end of the sentence: these indices are captured in st_all; returns start positions
                       if i!=0 and i+len(h) != len(dat): #The hate word/phrase starts and ends in the middle of the string
                           if dat[i-1] == " " and dat[i+len(h)] == " ": #To prevent capturing hate phrases that start from the middle of a word and end in the middle of a word 
                               st_all.append(i)
                       elif i!=0:
                           if dat[i-1] == " " and (i+len(h)) == len(dat): #The hate word/phrase starts in the middle and end at the string end
                               st_all.append(i)
                       elif i==0 and i+len(h) != len(dat): #The hate word/phrase starts at the beginning and ends in the middle
                           if dat[i+len(h)] == " ":
                               st_all.append(i)
                       elif i==0 and i+len(h) == len(dat): #The hate word/phrase starts in the beginning and end at the end of the string, eg. for sentences with single word or few words
                             st_all.append(i)

               for j in st_all:
                   temp_str = temp_str[:j]+"S"+"O"*(len(h)-1)+temp_str[j+len(h):] #Marking all characters that are part of OTG with either S or O (start of a segment is marked with S), so that if two consecuive words marked OTG are part of different segments, they can be distinguished; can't do this at the word level as muliple words could be part of the same hate lexicon
       ind = 0
       while ind < len(dat):
          if temp_str[ind] == "O" or temp_str[ind] == "S":  #words in the hate-phrase hate word/phrase  #The positions searched by 'ind' will always be starting positions of words as 'ind' gets updated in the next lines such that the starting index of the next word is assigned to 'ind'
              if "N" in temp_str[ind:]: #The hate word/phrase ends in the middle of a sentence 
                 if temp_str[ind-1] == "N": #Checking if the word is the start of a tagged segment
                        f.write(dat[ind:dat.find(" ",ind)]+" B-OTG"+"\n")
                 elif temp_str[ind] == "S":   #If previous word was OTG, but part of a different hate term
                        f.write(dat[ind:dat.find(" ",ind)]+" B-OTG"+"\n") 
                 else:
                        f.write(dat[ind:dat.find(" ",ind)]+" I-OTG"+"\n") #It means previous word was part of the OTG phrase containing multiple words 
                 ind = dat.find(" ",ind)+1 #Index of the next word after space
              elif " " in dat[ind:]: #The hate word/phrase traverses to the end of the sentence as there is no N, but has multiple words indicated by the presence of space
                  if temp_str[ind-1] == "N":
                      f.write(dat[ind:dat.find(" ",ind)]+" B-OTG"+"\n")
                  elif temp_str[ind] == "S":   #If previous word was OTG, but part of a different phrase
                        f.write(dat[ind:dat.find(" ",ind)]+" B-OTG"+"\n")
                  else:
                      f.write(dat[ind:dat.find(" ",ind)]+" I-OTG"+"\n")
                  ind = dat.find(" ",ind)+1 #Index of the next word after space
              else:
                  if temp_str[ind-1] == "N": 
                    f.write(dat[ind:]+" B-OTG"+"\n")
                  elif temp_str[ind] == "S":   #If previous word was OTG, but part of a different phrase
                        f.write(dat[ind:]+" B-OTG"+"\n")
                  else:
                        f.write(dat[ind:]+" I-OTG"+"\n")
                  ind = len(dat)   
          else:
              if dat.find(" ",ind) != -1:
                f.write(dat[ind:dat.find(" ",ind)]+" O"+"\n")
                ind = dat.find(" ",ind)+1
              else:
                  f.write(dat[ind:]+" O"+"\n")
                  ind = len(dat)
              
       f.write("\n")





   #----------------------

def get_target_lexs(file_tar):

    target_lex = pd.read_csv(file_tar ,sep="\t",header=0)

    targets = []

    targets.extend(target_lex.iloc[:,0].values)   

    target_terms = [tar.lower() for tar in targets if 'queer' not in tar.lower()]

    return target_terms



def prepare_iden_data_hard(inp_data, label_file, file_tar, out_file):


   with open(inp_data) as f:
     data = list(f)

   data = [tx[:-1].lower() for tx in data]

   with open(label_file) as f:
      lab = list(f)

   labels = [int(l[:-1]) for l in lab]

   data_hate = []
   for ind in range(len(labels)):
     if labels[ind] == 1:
       data_hate.append(data[ind])  #Only hateful instances used


   targets = get_target_lexs(file_tar)
   targets = list(set(targets))
#   written_wrds = []
   ind_sen = 0
   with open(out_file,'w') as f:
     for dat in data_hate:  
       ind_sen+=1
       tar_ind = []
       temp_str = "N"*len(dat) #This string keeps track of the positions that should be tagged as OTG
       for h in targets:
           if h in dat:
               st_all = []
               for i in range(len(dat)):
                   if dat.startswith(h,i): #To obtain all positions in the sentence where the hate term occurs from start of a word and ends at the end position of a word or at the end of the sentence: these indices are captured in st_all; returns start positions
                       if i!=0 and i+len(h) != len(dat): #The hate word/phrase starts and ends in the middle of the string
                           if dat[i-1] == " " and dat[i+len(h)] == " ": #To prevent capturing hate phrases that start from the middle of a word and end in the middle of a word 
                               st_all.append(i)
                       elif i!=0:
                           if dat[i-1] == " " and (i+len(h)) == len(dat): #The hate word/phrase starts in the middle and end at the string end
                               st_all.append(i)
                       elif i==0 and i+len(h) != len(dat): #The hate word/phrase starts at the beginning and ends in the middle
                           if dat[i+len(h)] == " ":
                               st_all.append(i)
                       elif i==0 and i+len(h) == len(dat): #The hate word/phrase starts in the beginning and end at the end of the string, eg. for sentences with single word or few words
                             st_all.append(i)

               for j in st_all:
                   temp_str = temp_str[:j]+"S"+"O"*(len(h)-1)+temp_str[j+len(h):] #Marking all characters that are part of OTG with either S or O (start of a segment is marked with S), so that if two consecuive words marked OTG are part of different segments, they can be distinguished; can't do this at the word level as muliple words could be part of the same hate lexicon
       ind = 0
       while ind < len(dat):
          if temp_str[ind] == "O" or temp_str[ind] == "S":  #words in the hate-phrase hate word/phrase  #The positions searched by 'ind' will always be starting positions of words as 'ind' gets updated in the next lines such that the starting index of the next word is assigned to 'ind'
              if "N" in temp_str[ind:]: #The hate word/phrase ends in the middle of a sentence 
                 if temp_str[ind-1] == "N": #Checking if the word is the start of a tagged segment
                        f.write(dat[ind:dat.find(" ",ind)]+" B-IDEN"+"\n")
                 elif temp_str[ind] == "S":   #If previous word was OTG, but part of a different hate term
                        f.write(dat[ind:dat.find(" ",ind)]+" B-IDEN"+"\n") 
                 else:
                        f.write(dat[ind:dat.find(" ",ind)]+" I-IDEN"+"\n") #It means previous word was part of the OTG phrase containing multiple words 
                 ind = dat.find(" ",ind)+1 #Index of the next word after space
              elif " " in dat[ind:]: #The hate word/phrase traverses to the end of the sentence as there is no N, but has multiple words indicated by the presence of space
                  if temp_str[ind-1] == "N":
                      f.write(dat[ind:dat.find(" ",ind)]+" B-IDEN"+"\n")
                  elif temp_str[ind] == "S":   #If previous word was OTG, but part of a different phrase
                        f.write(dat[ind:dat.find(" ",ind)]+" B-IDEN"+"\n")
                  else:
                      f.write(dat[ind:dat.find(" ",ind)]+" I-IDEN"+"\n")
                  ind = dat.find(" ",ind)+1 #Index of the next word after space
              else:
                  if temp_str[ind-1] == "N": 
                    f.write(dat[ind:]+" B-IDEN"+"\n")
                  elif temp_str[ind] == "S":   #If previous word was OTG, but part of a different phrase
                        f.write(dat[ind:]+" B-IDEN"+"\n")
                  else:
                        f.write(dat[ind:]+" I-IDEN"+"\n")
                  ind = len(dat)   
          else:
              if dat.find(" ",ind) != -1:
                f.write(dat[ind:dat.find(" ",ind)]+" O"+"\n")
                ind = dat.find(" ",ind)+1
              else:
                  f.write(dat[ind:]+" O"+"\n")
                  ind = len(dat)
              
              
       f.write("\n")



#-----------------------------------------------

def prepare_data_iden_soft_otg_hard(inp_data, label_file, file_hb, file_tar, out_file):
    """
    first tag with hard-mathed otg, then overwrite hard-matched iden tagging
    """
    prepare_otg_data(inp_data, label_file, file_hb, 'temp_file.txt')
    with open("temp_file.txt", "r") as file:
        lines = file.readlines()
    idens = get_target_lexs(file_tar)
    new_lines = []
    pbar = tqdm(total = 100)
    for iden in idens:
        for ind in range(1,len(lines)):
            if iden in lines[ind]:
                if "B-IDEN" in lines[ind -1]:
                    lines[ind] = lines[ind].replace(" O", " I-IDEN", 1)
                    
                elif "B-IDEN" not in lines[ind -1]:
                    lines[ind] = lines[ind].replace(" O", " B-IDEN", 1)
        pbar.update()
                
    with open('temp1.txt') as file:
        non_hate_lines = list(file.readlines())
    with open(out_file, "w") as file:
        for nonh_line in non_hate_lines:
            file.write(nonh_line)
        for line in lines:
            file.write(line)
    
    

    import os
    os.remove("temp_file.txt")
    os.remove('temp1.txt')   



def prepare_data_iden_soft_otg_hard_idenall(inp_data, label_file, file_hb, file_tar, out_file):
    """
    first tag with hard-mathed otg, then overwrite soft-matched iden tagging
    """
    prepare_otg_data(inp_data, label_file, file_hb, 'temp_file.txt')
    with open("temp_file.txt", "r") as file:
        lines = file.readlines()

    with open('temp1.txt') as file:
        non_hate_lines = list(file.readlines())

    lines = non_hate_lines + lines
    idens = get_target_lexs(file_tar)
    new_lines = []
    pbar = tqdm(total = 100)
    for iden in idens:
        for ind in range(1,len(lines)):
            if iden in lines[ind]:
                if "B-IDEN" in lines[ind -1]:
                    lines[ind] = lines[ind].replace(" O", " I-IDEN", 1)
                    
                elif "B-IDEN" not in lines[ind -1]:
                    lines[ind] = lines[ind].replace(" O", " B-IDEN", 1)
        pbar.update()
                
    with open(out_file, "w") as file:
        for line in lines:
            file.write(line)

    import os
    os.remove("temp_file.txt")
    os.remove('temp1.txt')        

        
def prepare_data_iden_soft_otg_hard_tagall(inp_data, label_file, file_hb, file_tar, out_file):
    """
    first tag with hard-mathed otg, then overwrite soft-matched iden tagging
    """
    prepare_otg_data(inp_data, label_file, file_hb, 'temp_file.txt')
    with open("temp_file.txt", "r") as file:
        lines = file.readlines()

    with open('temp1.txt') as file:
        non_hate_lines = list(file.readlines())

    lines = non_hate_lines + lines
    idens = get_target_lexs(file_tar)
    new_lines = []
    pbar = tqdm(total = 100)
    for iden in idens:
        for ind in range(1,len(lines)):
            if iden in lines[ind]:
                if "B-IDEN" in lines[ind -1]:
                    lines[ind] = lines[ind].replace(" O", " I-IDEN", 1)
                    
                elif "B-IDEN" not in lines[ind -1]:
                    lines[ind] = lines[ind].replace(" O", " B-IDEN", 1)
        pbar.update()

    otgs = get_hate_base_lexs(file_hb)
    for otg in otgs:
        for ind in range(1,len(lines)):
            if otg == lines[ind].split(' ')[0]:
                if "B-OTG" in lines[ind -1]:
                    lines[ind] = lines[ind].replace(" O", " I-OTG", 1)
                    
                elif "B-OTG" not in lines[ind -1]:
                    lines[ind] = lines[ind].replace(" O", " B-OTG", 1)
                
    with open(out_file, "w") as file:
        for line in lines:
            file.write(line)

    import os
    os.remove("temp_file.txt")
    os.remove('temp1.txt')   
                


