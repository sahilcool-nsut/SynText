from IPython.display import clear_output 
clear_output()

import numpy as np
import pandas as pd
import contractions
from emot.emo_unicode import EMOTICONS_EMO # For EMOTICONS
import emoji
import math
import json
import random
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
import torch
import tensorflow as tf
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import datasets
from datasets import load_dataset
import lexsub
from lexsub import BERTLexSub, LexSubUtils, LexSubExample
import torch
from typing import List
from transformers import *
import nltk
from nltk.corpus import stopwords
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pickle
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.test.utils import common_texts
import spacy

clear_output()

cols=["text",'label',"id"]

train_dataset=pd.read_csv('/kaggle/input/goemotionsfinal/train.tsv', sep='\t',names=cols, header=None)

test_dataset=pd.read_csv('/kaggle/input/goemotionsfinal/test.tsv', sep='\t',names=cols, header=None)

validation_dataset=pd.read_csv('/kaggle/input/goemotionsfinal/dev.tsv', sep='\t',names=cols, header=None)


import pandas as pd

emotions=[ 'admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']
dictionary_not_op={}
for j in train_dataset['label']:
    if j in dictionary_not_op.keys():
        dictionary_not_op[j]+=1
    else:
        dictionary_not_op[j]=1
for i in range(28):
    if(str(i) in dictionary_not_op.keys()):
        print(emotions[i],"\t\t",dictionary_not_op[str(i)])
len(train_dataset)

# # Ekman

# cols=["text","id","neutral","anger_","disgust_","fear_","joy_","sadness_","surprise_","label"]

# train_dataset=pd.read_csv('/kaggle/input/goemotionsekmann/training_ekman.csv', sep=',',names=cols, header=None)

# test_dataset=pd.read_csv('/kaggle/input/goemotionsekmann/testing_ekman.csv', sep=',',names=cols, header=None)

# validation_dataset=pd.read_csv('/kaggle/input/goemotionsekmann/validation_ekman.csv', sep=',',names=cols, header=None)

# train_dataset = pd.read_csv('/kaggle/input/ekman-augmented-new-final/ekman_augmented_new_final.csv', sep=',',names=cols, header=None)
# emotions=[ 'neutral','anger','disgust','fear','joy','sadness','surprise']
# dictionary_not_op={}
# for j in train_dataset['label']:
#     if j in dictionary_not_op.keys():
#         dictionary_not_op[j]+=1
#     else:
#         dictionary_not_op[j]=1

dictionary_op={}
for i in train_dataset['label']:
    i=i.split(',')
    for j in i:
        if j in dictionary_op.keys():
            dictionary_op[j]+=1
        else:
            dictionary_op[j]=1

print(dictionary_not_op)

for i in range(28):
    if(str(i) in dictionary_not_op.keys()):
        print(emotions[i],"\t\t",dictionary_not_op[str(i)])

class PreprocessingModule():
    def __init__(self):
        pass
    def extract_emoticons(self,s):
        EMOTICONS_EMO["<3"] = "Heart"
        newSentence=""
        for word in s.split():
            if word in EMOTICONS_EMO.keys():
                newSentence += " " + EMOTICONS_EMO[word] + " "
            else:
                newSentence += " " + word
        return newSentence.strip()
    def demojizeEmoji(self,s):
        demojized = emoji.demojize(s,delimiters=("",""))
        splitString = demojized.split('_')
        return " ".join(splitString)
    def replaceEmojis(self,dataset):
        # emot = required for emoticons
        # emoji = 1.6k stars, hence using this isntead of demoji

        dataset['text'] = dataset['text'].apply(lambda x: self.extract_emoticons(x))
        dataset['text'] = dataset['text'].apply(lambda x: self.demojizeEmoji(x))
    def mapContractions(self,dataset):
        # To Lower case AND contraction mapped
        dataset['text'] = dataset['text'].apply(lambda x: contractions.fix(x.lower()))

preprocesser = PreprocessingModule()

#preprocessing done here
preprocesser.mapContractions(train_dataset)
preprocesser.replaceEmojis(train_dataset)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
model_masked = BertForMaskedLM.from_pretrained("bert-base-uncased")
model_bert = model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
clear_output()

def convert_tokens_to_ids(tokens, tokenizer):
    """Converts tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        token_id = tokenizer._convert_token_to_id(token)
        ids.append(token_id)
    return ids

def rev_wordpiece(str):
    """wordpiece function used in cbert"""
    no_words=['!','@','#','$','%','^','&','*','(',')','_','-','=','+','/','*','[',']','{','}',':',';','"','<','>','?',',','.','`','|',"'",'...',' ']
    #print(str)
    n_list=len(str)
    if len(str) > 1:
        for i in range(len(str)-1, -1, -1):
#             print(str[i]," at -->  ",i)
            if str[i] == '[PAD]':
                str.remove(str[i])
            elif str[i] == '[UNK]':
                str.remove(str[i])
            elif len(str[i])<1 or (str[i] in no_words):
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#' and (len(str)-(n_list-i))==0:
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return " ".join(str)#[1:-1]

def convert_ids_to_str(ids, tokenizer):
    """converts token_ids into str."""
    tokens = []
    for token_id in ids:
        token = tokenizer._convert_id_to_token(token_id)
        tokens.append(token)
    outputs = rev_wordpiece(tokens) # If we dont need the preprocessing then simply return tokens
    return outputs

def synonym_op(text, tokenizer, model, number_of_synonym=10):

    tokens_a = tokenizer._tokenize(text)
    tokens = []
    tokens.append('[CLS]')
    for token in tokens_a:
        tokens.append(token)
    tokens.append('[SEP]')
    init_ids = convert_tokens_to_ids(tokens, tokenizer)
    segment_id = len(init_ids) * [1]

    init_ids = torch.tensor([init_ids])
    segment_id = torch.tensor([segment_id])
    masked_idx=[]
    for i in range(len(init_ids)):
        masked_idx.append(i)

    #here are the predictions

    predictions = model(init_ids, segment_id)
    predictions = torch.nn.functional.softmax(predictions[0], dim=2)

    synonym_list=[]
    all_word_predictions=torch.multinomial(predictions[0], number_of_synonym)
    for i in range(len(tokens)):
        preds = all_word_predictions[i]
        new_str = convert_ids_to_str(preds.numpy(), tokenizer)
        synonym_list.append(new_str.split(' '))
    return tokens, synonym_list # it returns token and synonym of that token in synonym_list

nltk.download('stopwords')
other_stop_words="a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours	ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves . , ; & ^ % $ # @ ! ( ) / * - + ? > < ' [ ] { } _ - + = ~ ` ing ed eg full less able ous ic ive ant ly ion ness ment ity er eer en ize ism ist ty ity or ship sion tion [cls] [sep]"
sw_nltk = set(stopwords.words('english'))
other_stop_words=set(other_stop_words.split(" "))
sw_nltk=sw_nltk.union(other_stop_words)

def cosine_similarity(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim


def data_split(data):
    deli = ['!','@','#','$','%','^','&','*','(',')','[',']','{','}','-','_','+','=',':',';','"',"'",',','<','.','>']
    new_data=""
    for c in data:
        if c not in deli:
              new_data+=c
    return new_data.split(" ")

def sortSecond(val):
    return val[1]

def embed(text, tokenizer, model):

    tokenized_text = tokenizer._tokenize("[CLS] " + text + " [SEP]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    with torch.no_grad():
        # uncomment prints to understand dimensions
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
#         print(token_embeddings.size())
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
#         print(token_embeddings.size())
        token_embeddings = token_embeddings.permute(1,0,2)
#         print(token_embeddings.size())
        token_vecs_sum = []
        for token in token_embeddings:
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)
#         print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
        return token_vecs_sum,tokenized_text
#         return hidden_states,tokenized_text

class ScoringModule():
    def __init__(self):
        self.model_word2vec = FastText(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
        pass
    def calculate_scores(self,token,synonyms_list,text,token_index,token_embedding,tokenizer_bert,model_bert):
        score_dict={}
        for synonym in synonyms_list:
            currentScore = 0
            # Currently equal weights (1/2) given to each score
            wordSimilarityContextual = self.word_to_word_similarity_contextual(text,token_index,token_embedding,token,synonym,tokenizer_bert,model_bert)
#             print(wordSimilarityContextual)
#             print("Contextual Similarity of " + token + " and " + synonym + " is: " + str(wordSimilarityContextual))
            currentScore+=wordSimilarityContextual
            wordSimilarity = self.word_to_word_similarity_non_contextual(token,synonym,tokenizer_bert,model_bert)
#             print("Non Contextual Similarity of " + token + " and " + synonym + " is: " + str(wordSimilarity))
            
            currentScore+=wordSimilarity
            
            currentScore = currentScore/2
#             print("Total score of " + token + " and " + synonym + " is: " + str(currentScore))
#             print(synonym + ": " + str(currentScore))
    
#             print(synonym + ": " + str(wordSimilarity))
            if(currentScore >= 0.60):
                score_dict[synonym]=currentScore
            else:
                continue
        score_dict=sorted(score_dict.items(), key=lambda x:x[1],reverse=True)
#         print(score_dict)
        return score_dict
    def word_to_word_similarity_contextual(self,text,token_index,token_embedding,token,synonym,tokenizer_bert,model_bert):
        temp_augmented = text.replace(token, synonym)
        embeddings_text_temp_augmented,token_text_temp_augmented=embed(temp_augmented,tokenizer_bert, model_bert)

        augmented_similarity = cosine_similarity(token_embedding,embeddings_text_temp_augmented[token_index])

        return augmented_similarity

    def word_to_word_similarity_non_contextual(self,token,synonym,tokenizer_bert,model_bert):
        embeddings_token,token_text =embed(token,tokenizer_bert, model_bert)
        embeddings_synonym, token_text = embed(synonym,tokenizer_bert,model_bert)
        return cosine_similarity(embeddings_token[1],embeddings_synonym[1])
        
        

def probability(number_augmentation):
    if(random.random()<=number_augmentation):
        return True
    else:
        return False

def augment_data(text, text_label, cols, stop_words, number_augmentation_needed, tokenizer_bert, model_masked, model_bert, word_similarity_threshold=0.6, no_of_word_replacement_ratio=0.4):
#     Probability of each sentence to be augmented one extra time
    if probability(number_augmentation_needed - math.floor(number_augmentation_needed)):
        number_augmentation_needed+=1
    
    if(number_augmentation_needed<1):
        return "",text_label
    
    number_augmentation_needed = math.floor(number_augmentation_needed)
    scoring_module = ScoringModule()
    
   
    #  ------------------------------- WORD CHOOSING PART ---------------------------------------
    
    similarity_with_emotion=[]
    chosen_words = []
    
    
#     lets get the sentence embeddings using bert
    embeddings_text,token_text=embed(text,tokenizer_bert, model_bert)
    embeddings_class,token_class=embed(cols[int(text_label)],tokenizer_bert, model_bert)

    for i in range(len(token_text)):
        if(token_text[i].lower() in stop_words):
            continue
        if(token_text[i][0] == '[' and token_text[i][-1] ==']'):
            continue
        if token_text[i] not in data_split(text):
            continue
        
    # TO BE CHECKED
        currentTokenEmbedding = embeddings_text[i].numpy()

        new_sentence_embeddings,new_token_text = embed(text.replace(token_text[i],cols[int(text_label)]),tokenizer_bert,model_bert)
        emotionEmbedding = new_sentence_embeddings[i]
        similarity_value = cosine_similarity(currentTokenEmbedding,emotionEmbedding)

#         print(token_text[i] + " " + cols[int(text_label)] + " " + str(similarity_value) )
        
        if(similarity_value >= word_similarity_threshold):  
            similarity_with_emotion.append([token_text[i],similarity_value]) 
            chosen_words.append(token_text[i])
    
    
        
    similarity_with_emotion.sort(key=sortSecond, reverse=True)
#     print(len(token_text))
#     print(similarity_with_emotion)
#     print("Length: " + str(len(similarity_with_emotion)))
#     print("Length of word choosing: " + str(len(chosen_words)))
    
    no_of_replacement=math.ceil(no_of_word_replacement_ratio*len(similarity_with_emotion))
#     print("Number of replacements: " + str(no_of_replacement))
    
    #  ------------------------------- SYNONYM COLLECTION PART ---------------------------------------
        
    token_indexes={}
    index_of_token=0

    tokens,synonym_tokens=synonym_op(text, tokenizer_bert, model_masked, number_augmentation_needed*2)#  getting twice the number of synonym of the augmented number of sentences, but we may get less based on the preprocessing of those synonyms
    # Ye banai hai taki baad mai mai token se us ka index le lu and then synonym_token list ko access kr saku for that token
    for token in tokens:
        token_indexes[token]=index_of_token
        index_of_token+=1
  
#     print(tokens)
#     print(synonym_tokens)

    #  ------------------------------- SCORING PART ---------------------------------------
    score_dictionary={}
    
    for i in range(0,len(tokens)):
        current_token = tokens[i]
        if(current_token not in chosen_words):
            continue
        if(current_token[0] == '[' and current_token[-1] ==']'):
            continue
        # calculate_score returns a dictionary sorted with each synonym's score
        score_dictionary[current_token] = scoring_module.calculate_scores(current_token,synonym_tokens[i],text,i,embeddings_text[i],tokenizer_bert,model_bert)
#         print(current_token)
#         print(score_dictionary[current_token])
    new_syn_tok =[]
    for i in range(0,len(tokens)):
        current_token=tokens[i]
        temp_list=[]
        if current_token in score_dictionary:
            for item in score_dictionary[current_token]:
                temp_list.append(item[0])
        new_syn_tok.append(temp_list)
    synonym_tokens=new_syn_tok
    
#     print(similarity_with_emotion)
#     print("Length: " + len(similarity_with_emotion))
#     print("Length of word choosing: " + len(chosen_words))   
   
    
    augmented_data=[]
    augmented_data_labels=[]
    for i in range(number_augmentation_needed):
        augmented_sentence=text
        length_n=len(similarity_with_emotion)
        
        for word in similarity_with_emotion:
            word=word[0]

            if(word not in token_indexes.keys()):
                continue
            
            temp_augmented=augmented_sentence
            if(len(synonym_tokens[token_indexes[word]])<=i):
                continue
            temp_augmented = temp_augmented.replace(word, synonym_tokens[token_indexes[word]][i])
            
            augmented_sentence=temp_augmented
        
        if(augmented_sentence==text):
            continue
        augmented_data.append(augmented_sentence)
        augmented_data_labels.append(text_label)
#         print("augmented_sentence : ",augmented_sentence)
#         print("sentence augmented---------------------------------------------------")
    
    return augmented_data, augmented_data_labels

# Uncomment this for single data testing..........................................................

# text="Happy to be able to help"
# text="The crowd gathered outside the house. I was angry that he had forgotten my birthday."
# text="Also available on Spotify and Apple Music. Any feedback would be hugely appreciated!"
# text="It might be linked to the trust factor of your friend."
# text="All sounds possible except the key, I can't see how it was missed in the first search."
# text="Man'. The point became invalid once the school yard name calling came out. And, 'this year'. It's only January."
# text="you are a disgusting piece of filth."
# text="I read on a different post that he died shortly after of internal injuries "
# text="Imagine if we dont trade [NAME] and [NAME] fans just kill all of us."
# text="also anxious that people will be angry or surprised or upset at me for me never telling people about this before"
# text="omg pizza time with junko time"

# x,y=augment_data(text, '0',cols,sw_nltk, 10, tokenizer, model_masked, model_bert,0.6, 0.3)

# print(x)
# print(y)

import warnings
import time

start_time=0
end_time=0
warnings.filterwarnings("ignore")
train_augmented_data=[]
train_augmented_labels=[]
i=-1
errorCounter = 0

while i<len(train_dataset["text"])-1 :
    try:
        i=i+1
        #     print(i)
        if(i%200==0):
            clear_output()
            print(i/len(train_dataset["text"]),"%")
            start_time = end_time
            end_time = time.time()
            print(end_time-start_time)

        min_data_points=800
        train_augmented_data.append(train_dataset['text'][i])
        train_augmented_labels.append(train_dataset['label'][i])
        ratio=min_data_points/dictionary_not_op[train_dataset['label'][i]]

        if(len((train_dataset['label'][i]).split(','))>1):
            continue

        if(dictionary_not_op[train_dataset['label'][i]]>min_data_points):
            ratio = 500/dictionary_not_op[train_dataset['label'][i]]

        x,y=augment_data(train_dataset['text'][i], train_dataset['label'][i],cols,sw_nltk, ratio, tokenizer, model_masked, model_bert,0.6, 0.3)
        if(len(x)>0):
            train_augmented_data.extend(x)
            train_augmented_labels.extend(y)
        if(i>20000):
            break
    except:
        print("Error")
        print(train_dataset["text"][i])
        errorCounter+=1
print(len(train_augmented_data))
print(len(train_augmented_labels))
print(errorCounter)
print(i)
df=[]
df = pd.DataFrame(train_augmented_data, columns =['text'])
df['label']=train_augmented_labels
df.to_csv('/kaggle/working/aug_data_new_20k.csv',index=False)

print(len(train_dataset["text"]))

dictionary_not_op_aug={}
for j in df['label']:
    if j in dictionary_not_op_aug.keys():
        dictionary_not_op_aug[j]+=1
    else:
        dictionary_not_op_aug[j]=1
