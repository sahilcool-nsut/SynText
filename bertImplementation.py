import pandas as pd

clear_output()

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AutoTokenizer, DebertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix, confusion_matrix
from transformers import EvalPrediction
import torch
import tensorflow as tf
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import datasets
import contractions
from emot.emo_unicode import EMOTICONS_EMO # For EMOTICONS
import emoji

# AUGMENTED EKMAN
# cols=["text",'label']
# train_dataset=pd.read_csv('/kaggle/input/d/sahilchawla7/ekman-augmented-new-final/ekman_augmented_new_final.csv', sep=',',names=cols, header=None)
# # train_dataset2=pd.read_csv('/kaggle/input/ekmanaugmentedprobability/aug_data_new_2_final.csv', sep=',',names=cols, header=None)
# # train_dataset = pd.concat([train_dataset1,train_dataset2],axis=0, ignore_index=True)
# cols_test = ["text","id","neutral","anger_","disgust_","fear_","joy_","sadness_","surprise_","label"]
# test_dataset=pd.read_csv('/kaggle/input/goemotionsekmann/testing_ekman.csv', sep=',',names=cols_test,header=None)
# validation_dataset=pd.read_csv('/kaggle/input/goemotionsekmann/validation_ekman.csv', sep=',',names=cols_test, header=None)

# UNAUGMENTED EKMAN
# cols_test = ["text","id","neutral","anger_","disgust_","fear_","joy_","sadness_","surprise_","label"]
# train_dataset=pd.read_csv('/kaggle/input/goemotionsekmann/training_ekman.csv', sep=',',names=cols_test,header=None)
# test_dataset=pd.read_csv('/kaggle/input/goemotionsekmann/testing_ekman.csv', sep=',',names=cols_test,header=None)
# validation_dataset=pd.read_csv('/kaggle/input/goemotionsekmann/validation_ekman.csv', sep=',',names=cols_test, header=None)

# # Fine Grained
cols_test = ["text","label","id"]
train_dataset=pd.read_csv('/kaggle/input/goemotionsfinal/train.tsv', sep='\t',names=cols_test,header=None)
test_dataset=pd.read_csv('/kaggle/input/goemotionsfinal/test.tsv', sep='\t',names=cols_test,header=None)
validation_dataset=pd.read_csv('/kaggle/input/goemotionsfinal/dev.tsv', sep='\t',names=cols_test, header=None)

train_dataset = train_dataset[['text','label']].copy()
test_dataset = test_dataset[['text', 'label']].copy()
validation_dataset = validation_dataset[['text', 'label']].copy()
train_dataset

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
        dataset['text'] = dataset['text'].apply(lambda x: contractions.fix(x))

preprocesser = PreprocessingModule()

#preprocessing done here
preprocesser.mapContractions(train_dataset)
preprocesser.replaceEmojis(train_dataset)
preprocesser.mapContractions(test_dataset)
preprocesser.replaceEmojis(test_dataset)
preprocesser.mapContractions(validation_dataset)
preprocesser.replaceEmojis(validation_dataset)


#  Only for Ekman !!
# train_dataset.drop(0,inplace=True)
# test_dataset.drop(0,inplace=True)
# validation_dataset.drop(0,inplace=True)

# # One-Hot Encoding

labels_train=train_dataset['label']
labels_train=labels_train.to_numpy()
labels_train_integer=[]
for entry in labels_train:
    ent=entry.split(',')
    labels_train_integer.append(np.array(ent,dtype=int))
mlb = MultiLabelBinarizer()
new_labels_train=mlb.fit_transform(labels_train_integer)

labels_test=test_dataset['label']
labels_test=labels_test.to_numpy()
labels_test_integer=[]
for entry in labels_test:
    ent=entry.split(',')
    labels_test_integer.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_test=mlb.fit_transform(labels_test_integer)

labels_valid=validation_dataset['label']
labels_valid=labels_valid.to_numpy()
labels_valid_integer=[]
for entry in labels_valid:
    ent=entry.split(',')
    labels_valid_integer.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_valid=mlb.fit_transform(labels_valid_integer)

train_dataset.drop('label',axis=1,inplace=True)
test_dataset.drop('label',axis=1,inplace=True)
validation_dataset.drop('label',axis=1,inplace=True)


# # Setting Dataset into a certain format

# fine grained
emotion_labels=[ 'admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']
# for ekman
# emotion_labels = ["neutral","anger","disgust","fear","joy","sadness",'surprise']
labels_train_one_hot=pd.DataFrame(new_labels_train,columns=emotion_labels)
labels_test_one_hot=pd.DataFrame(new_labels_test,columns=emotion_labels)
labels_valid_one_hot=pd.DataFrame(new_labels_valid,columns=emotion_labels)

# Just to check if the one hot encoding was done correctly

# pd.set_option('display.max_columns', None)

# print(train_dataset.head(20))
# labels_train_one_hot.head(20)
# train_dataset,labels_train_one_hot

training_data_final=pd.concat([train_dataset.reset_index(),labels_train_one_hot],axis=1)
testing_data_final=pd.concat([test_dataset.reset_index(),labels_test_one_hot],axis=1)
validation_data_final=pd.concat([validation_dataset.reset_index(),labels_valid_one_hot],axis=1)

training_data_final.drop('index',axis=1,inplace=True)
testing_data_final.drop('index',axis=1,inplace=True)
validation_data_final.drop('index',axis=1,inplace=True)

hg_dataset_train = Dataset(pa.Table.from_pandas(training_data_final))
hg_dataset_test = Dataset(pa.Table.from_pandas(testing_data_final))
hg_dataset_valid = Dataset(pa.Table.from_pandas(validation_data_final))

dataset={}
dataset['train']=hg_dataset_train
dataset['test']=hg_dataset_test
dataset['validation']=hg_dataset_valid

dataset = datasets.DatasetDict(dataset)

labels = [label for label in dataset['train'].features.keys() if label not in ['text', 'id','index']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels,id2label,label2id

# # Encoding Process

# Bert case
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

# # Example

# %% [code] {"execution":{"iopub.status.busy":"2023-04-09T04:10:11.664617Z","iopub.execute_input":"2023-04-09T04:10:11.664963Z","iopub.status.idle":"2023-04-09T04:10:11.670811Z","shell.execute_reply.started":"2023-04-09T04:10:11.664933Z","shell.execute_reply":"2023-04-09T04:10:11.669975Z"}}
example = encoded_dataset['train'][0]
print(example)

tokenizer.decode(example['input_ids'])

# ### Converting into format for torch
# 

encoded_dataset.set_format("torch")

# # Model Definition

model = BertForSequenceClassification.from_pretrained("bert-base-cased",
                                                           problem_type="multi_label_classification", 
                                                           num_labels=28,
                                                           id2label=id2label,
                                                           label2id=label2id,
                                                           hidden_dropout_prob = 0.1
                                                          )

batch_size = 16

args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=4,#4
    weight_decay=0.01,
    warmup_ratio=0.01,
    metric_for_best_model="roc_auc",
    load_best_model_at_end=True,
    greater_is_better=True
)

def multi_label_metrics(predictions, labels, threshold=0.3):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs > threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    roc_auc = roc_auc_score(y_true, y_pred, average = 'macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    # return as dictionary
    metrics = {'f1_macro': f1_macro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy,
               'f1' : json.dumps(f1.tolist()),
               'Precision' : json.dumps(precision.tolist()),
               'Recall' : json.dumps(recall.tolist()),
              }
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

# #forward pass
# outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
# outputs

# %% [code]


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# API Key
#1daa61ad9ecca990d13db89f54ae5da1d6d0e458

model.eval()
preds = trainer.predict(encoded_dataset['test'])
print(preds)
