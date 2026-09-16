"""Track B: full SST sentiment classification, optional predicted-gaze fusion.

Personal experiment script (see docs/04 and docs/05). This is the *large-n*
track: 11,853 SST sentences with a 5-D *predicted* gaze vector
(nFix, FFD, GPT, TRT, GD), not lab-recorded eye tracking.

Switch `model_type` to compare text-only vs late fusion:

    bert | roberta | bert_eye_tracking | roberta_eye_tracking

The fusion graph is late: RoBERTa/BERT pooler (768) + Linear(5, 16),
concat, dropout 0.1, Linear(784, 3). Hyperparameters below are the
ones used for the hold-out run (5 epochs, batch 256, Adam 5e-5).

Known traps, left as-is so this file still matches the original
experiment:

- The test loop *assigns* `all_preds` instead of extending it, so the
  printed test metrics are the last batch only. Validation uses extend
  and is fine.
- `best_val_acc` is validation *accuracy*. The nearby comment/print
  still says F1.
- `models/` is not created for you; `torch.save` will fail if missing.
- The 80/10/10 CSVs come from `SST_data/spilt.py` (typo) and are not
  stratified.

For the 400-sentence real-gaze track, use `model_ZuCo_SST.py`.
"""

import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification

# --- run configuration -------------------------------------------------
# Five features, 16-D gaze tower, 3 SST classes. Batch 256 is the
# setting that wants a GPU; max_length is 128 and even the longest SST
# row in this clone is 56 words.
num_eye_tracking_features = 5
hidden_layer_size = 16
num_labels = 3
num_epochs = 5
learning_rate = 5e-5
batch_size = 256
model_type = 'roberta_eye_tracking'         # 'bert', 'bert_eye_tracking', 'roberta', 'roberta_eye_tracking'

# 数据集路径
train_dataset_path = 'SST_data/train_full_sst.csv'
valid_dataset_path = 'SST_data/valid_full_sst.csv'
test_dataset_path = 'SST_data/test_full_sst.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
print("Model type:", model_type)
# 初始化分词器
if model_type.startswith('bert'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
else:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

# Late-fusion classifier. The gaze Linear has no activation: it is an
# affine remix of five z-scored (predicted) reading-time features.
# Dropout sits *after* the concat, so it can zero text dims too.
# Both BERT and RoBERTa are read via `pooler_output` — see docs/04.
class EyeTrackingModel(nn.Module):
    def __init__(self, base_model, num_eye_tracking_features, num_labels):
        super().__init__()
        self.base_model = base_model.from_pretrained('bert-base-uncased' if base_model == BertModel else 'roberta-base')
        self.eye_tracking_layer = nn.Linear(num_eye_tracking_features, hidden_layer_size)
        self.classifier = nn.Linear(self.base_model.config.hidden_size + hidden_layer_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, eye_tracking_features):
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = base_output.pooler_output
        eye_tracking_output = self.eye_tracking_layer(eye_tracking_features)
        combined_output = torch.cat((pooled_output, eye_tracking_output), dim=1)
        combined_output = self.dropout(combined_output)
        return self.classifier(combined_output)

# 定义自定义的数据集类
class CustomDataset(TorchDataset):
    def __init__(self, dataframe, eye_tracking_features):
        self.encodings = {'input_ids': dataframe['input_ids'].tolist(),
                          'attention_mask': dataframe['attention_mask'].tolist()}
        self.labels = dataframe['sentiment_label'].values
        self.eye_tracking_features = eye_tracking_features

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['eye_tracking_features'] = torch.tensor(self.eye_tracking_features[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def calculate_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, precision, recall, f1

# Text-only branches ignore the gaze tensor that CustomDataset still
# yields. Fusion branches must keep num_labels == 3; the CE call below
# hard-codes logits.view(-1, 3).
def get_model(model_type, num_eye_tracking_features, num_labels):
    if model_type == 'bert':
        return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    elif model_type == 'roberta':
        return RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    elif model_type == 'bert_eye_tracking':
        return EyeTrackingModel(BertModel, num_eye_tracking_features, num_labels)
    elif model_type == 'roberta_eye_tracking':
        return EyeTrackingModel(RobertaModel, num_eye_tracking_features, num_labels)

# 加载数据集
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    eye_tracking_features = df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']]
    text_data = df[['sentence', 'sentiment_label']]
    hf_dataset = Dataset.from_pandas(text_data)
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    df_tokenized = tokenized_dataset.to_pandas()
    df_tokenized = pd.concat([df_tokenized, eye_tracking_features], axis=1)
    return CustomDataset(df_tokenized, eye_tracking_features.values)

train_dataset = load_dataset(train_dataset_path)
valid_dataset = load_dataset(valid_dataset_path)
test_dataset = load_dataset(test_dataset_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义模型
model = get_model(model_type, num_eye_tracking_features, num_labels)
model.to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)

# Despite the original comment, this is validation *accuracy*, not F1.
# torch.save will raise if the models/ directory does not exist.
best_val_acc = 0.0  # 初始化最佳F1分数
best_model_path = f'models/best_{model_type}_model.pth'  # 根据model_type动态构建最佳模型保存路径

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=(epoch == num_epochs - 1))
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if model_type == 'bert' or model_type == 'roberta':
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        else:
            eye_tracking_features_tensor = batch['eye_tracking_features'].to(device)
            logits = model(input_ids, attention_mask, eye_tracking_features_tensor)
            loss = CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"Training Loss": total_loss / len(train_loader)})

    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(valid_loader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if model_type == 'bert' or model_type == 'roberta':
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        else:
            eye_tracking_features_tensor = batch['eye_tracking_features'].to(device)
            logits = model(input_ids, attention_mask, eye_tracking_features_tensor)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算验证集上的指标
    val_acc, val_p, val_r, val_f1 = calculate_metrics(np.array(all_preds), np.array(all_labels))
    print(f"Validation Acc: {val_acc:.4f}, P: {val_p:.4f}, R: {val_r:.4f}, F1: {val_f1:.4f}")

    # 检查是否为最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved as {best_model_path} with F1: {best_val_acc:.4f}")

# 加载最佳模型
best_model = get_model(model_type, num_eye_tracking_features, num_labels)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)
model = best_model
# 在测试集上进行评估
model.eval()
all_preds = []
all_labels = []
for batch in tqdm(test_loader, desc="Testing"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    if model_type == 'bert' or model_type == 'roberta':
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    else:
        eye_tracking_features_tensor = batch['eye_tracking_features'].to(device)
        logits = model(input_ids, attention_mask, eye_tracking_features_tensor)

        preds = torch.argmax(logits, dim=1)
        # TRAP: assignment, not extend. Metrics below are last-batch only.
        # Validation a few dozen lines up correctly uses .extend(...).
        all_preds = preds.cpu().numpy()
        all_labels = labels.cpu().numpy()

# 计算测试指标
test_acc, test_p, test_r, test_f1 = calculate_metrics(np.array(all_preds), np.array(all_labels))
print("Model type:", model_type)
print(f"Test Acc: {test_acc:.4f}, P: {test_p:.4f}, R: {test_r:.4f}, F1: {test_f1:.4f}")