import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import pandas as pd
import numpy as np
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
torch.manual_seed(2025)
import logging
logging.basicConfig(level=logging.ERROR)
from sklearn.metrics import *
from torch import cuda
import re
device = 'cuda' if cuda.is_available() else 'cpu'
labels_dict={"Anti-EU (EU economic skepticism)":0,
              "Anti-EU (Crisis of EU)":1,
              "Anti-EU (EU political interference)":2,
              "Anti-EU (EU Corruption)":3,
              "Political hate and polarisation (Pro far-left)":4,
              "Political hate and polarisation (Pro far-right)":5,
              "Political hate and polarisation (Anti-liberal)":6,
              "Political hate and polarisation (Anti-woke)":7,
              "Religion-related (Anti-Islam)":8,
              "Religion-related (Anti-Semitic conspiracy theories)":9,
              "Religion-related (Interference with states' affairs)":10,
              "Gender-related (Language-related)":11,
              "Gender-related (LGBTQ+-related)":12,
              "Ethnicity-related (Association to political affiliation)":13,
              "Ethnicity-related (Ethnic generalisation)":14,
              "Ethnicity-related (Ethnic offensive language)":15,
              "Ethnicity-related (Threat to population narratives)":16,
              "Migration-related (Migrants societal threat)":17,
              "Distrust in institutions (Failed state)":18,
              "Distrust in institutions (Criticism of national policies)":19,
              "Distrust in democratic system (Elections are rigged)":20,
              "Distrust in democratic system (Anti-Political system)":21,
              "Distrust in democratic system (Anti-Media)":22,
              "Geopolitics (Pro-Russia)":23,
              "Geopolitics (Foreign interference)":24,
              "Geopolitics (Anti-international institutions)":25,
              "Anti-Elites (Soros)":26,
              "Anti-Elites (World Economic Forum / Great Reset)":27,
              "Anti-Elites (Green Agenda)":28,
              "None":29,
              "Anti-Elites (Antisemitism)":30,
              "Gender-related (Demographic narratives)":31,
              "Distrust in democratic system (Immigrants right to vote)":32  }

######################clean tweets######################################
def clean(text):
    text = re.sub(r"http\S+", " ", text) # remove urls
    text = re.sub(r"RT ", " ", text) # remove rt
    text = re.sub(r'\t', ' ', text) # remove tabs
    text = re.sub(r'\n', ' ', text) # remove line jump
    text = re.sub(r"\s+", " ", text) # remove extra white space 
    return text       

def calculate_accuracy(preds, targets):
    print(preds)
    n_correct = (preds==targets).sum().item()
    return n_correct
def calculate_MacroF1(preds,targets):
    return f1_score(targets,preds, average='macro')

def calculate_MicroF1(preds,targets):
    return f1_score(targets,preds, average='micro')

def calculate_MacroPrec(preds,targets):
    return precision_score(targets,preds, average='macro')   

def calculate_MacroRecall(preds,targets):
    return recall_score(targets,preds, average='macro')    

train = pd.read_csv('data/train.tsv', delimiter='\t')
train['label']=train['label'].fillna('None')
train_data = train[['tweet', 'label']]
train_data['tweet']=train_data['tweet'].apply(clean)
valid = pd.read_csv('data/valid.tsv', delimiter='\t')
valid['label']=valid['label'].fillna('None')
valid_data = valid[['tweet', 'label']]
valid_data['tweet']=valid_data['tweet'].apply(clean)


train_data= train_data.replace({"label": labels_dict})
valid_data= valid_data.replace({"label": labels_dict})

MAX_LEN = 200

TRAIN_BATCH_SIZE = 16

VALID_BATCH_SIZE = 16

# LEARNING_RATE = 1e-5
# LEARNING_RATE = 2e-5
LEARNING_RATE = 3e-5
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)


class NarrativesData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.tweet
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

print("TRAIN Dataset: {}".format(train_data.shape))
print("VALIDATION Dataset: {}".format(valid_data.shape))


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 4
                }

val_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 4
                }


training_set = NarrativesData(train_data, tokenizer, MAX_LEN)
validation_set = NarrativesData(valid_data, tokenizer, MAX_LEN)

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **val_params)


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-large")
        self.pre_classifier = torch.nn.Linear(1024, 1024)
    
        self.dropout = torch.nn.Dropout(0.3)
     
        self.classifier = torch.nn.Linear(1024, 30)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
        hidden_state = output_1[0]
     
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
       
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model.to(device)       

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE,weight_decay=1e-5)




def train(epoch):
   
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    
    model.train()

    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

       
        outputs = model(ids, mask, token_type_ids)

       
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()

       
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accuracy(big_idx, targets)

     
        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

       
        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

       
        optimizer.zero_grad()
        loss.backward()
      
        optimizer.step()
    
    
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return

EPOCHS = 5
for epoch in range(EPOCHS):
    train(epoch)    

torch.save(model.state_dict(),'roberta_models/roberta-large_%s_dropout0.3'%LEARNING_RATE )

def valid(model, validation_loader):

  
    model.eval()

    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0

    full_targets=[]
    full_pred=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()

           
            # loss = loss_function(outputs, targets)
            # tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            print(targets.tolist())
            print(big_idx.tolist())
            full_targets= full_targets+targets.tolist()
            full_pred=full_pred+big_idx.tolist()
            n_correct += calculate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

          
            if _%5000==0:
                # loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                # print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
                
 
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    # print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
   
    return epoch_accu, full_pred, full_targets

valid_acc, valid_pred,valid_targets  = valid(model, validation_loader)
print(len(valid_pred))
print(len(valid_targets ))
print("Macro F1 on dev set",calculate_MacroF1(valid_pred,valid_targets ))    
print("Micro F1 on dev set",calculate_MicroF1(valid_pred,valid_targets ))  
print("Macro Recall dev set",calculate_MacroRecall(valid_pred,valid_targets ))  
print("Macro Prec on dev set",calculate_MacroPrec(valid_pred,valid_targets ))     
