"""Track A: ZuCo 400-sentence sentiment + *real* averaged gaze, 5-fold CV.

Personal experiment script (see docs/01, docs/04, docs/05). This is the
small-n track: 400 movie-review sentences that were actually read in the
ZuCo lab by 12 people. Gaze columns are z-scored 12-reader means from
`ZuCo_SST_data/combined_sst_et_standard.csv`.

The 80/10/10 files `ZuCo_SST_data/{train,valid,test}.csv` are *not*
used here. Evaluation is `StratifiedKFold(n_splits=5, random_state=42)`.
Each fold builds a fresh model, trains 20 epochs, then scores the 80
held-out rows once. There is no inner validation / early stopping, and
no checkpoint is written.

`model_type` is the same four-way switch as Track B. Default is
`roberta_eye_tracking`. The fusion graph is identical: pooler 768 +
Linear(5, 16) → 784 → 3. Feature order must stay

    nFixations, FFD, GPT, TRT, GD

which is *not* the column order in the CSV (the CSV has omissionRate,
pupil, SFD in between). The explicit `df[[...]]` below is load-bearing.

For the full-SST / predicted-gaze track, use `model_full_SST.py`.
"""

import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from torch.utils.data import Dataset as TorchDataset

# --- run configuration -------------------------------------------------
# 20 epochs × 5 folds on ~320 sentences is long enough to overfit;
# report the mean of the five fold scores, not the best fold. batch_size
# is also hard-coded again when the DataLoaders are built (16).
num_eye_tracking_features = 5  # 例如：5
hidden_layer_size = 16         # 隐藏层大小
num_labels = 3                 # 标签数
num_epochs = 20                 # 训练轮数
learning_rate = 5e-5           # 学习率
batch_size = 16                # 批量大小
model_type = 'roberta_eye_tracking'            # 'bert', 'bert_eye_tracking', 'roberta', 'roberta_eye_tracking'

# 数据集路径
dataset_path = 'ZuCo_SST_data/combined_sst_et_standard.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
print("Model type:", model_type)

# Column pick is deliberate: eight numeric gaze-ish columns exist in
# the CSV; only these five enter the concat. See docs/01 for why SFD,
# omissionRate, and meanPupilSize stay out of the tower.
df = pd.read_csv(dataset_path)
eye_tracking_features = df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]

# 初始化分词器
if model_type.startswith('bert'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
else:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 分词函数
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

# 处理数据集
text_data = df[['sentence', 'sentiment_label']]
hf_dataset = Dataset.from_pandas(text_data)
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
df_tokenized = tokenized_dataset.to_pandas()
df_tokenized = pd.concat([df_tokenized, eye_tracking_features], axis=1)

# Same late-fusion module as model_full_SST.py. Duplicated on purpose
# so each track can be run as a single file. pooler_output is used for
# both BERT and RoBERTa (docs/04).
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

# 获取模型
def get_model(model_type, num_eye_tracking_features, num_labels):
    if model_type == 'bert':
        return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    elif model_type == 'roberta':
        return RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    elif model_type == 'bert_eye_tracking':
        return EyeTrackingModel(BertModel, num_eye_tracking_features, num_labels)
    elif model_type == 'roberta_eye_tracking':
        return EyeTrackingModel(RobertaModel, num_eye_tracking_features, num_labels)

# Five independent fine-tunes. fold scores are appended and averaged
# at the end; nothing is checkpointed. Stratify on sentiment_label so
# each 80-row test slice keeps the ~123/137/140 class mix.
val_accs = []
val_ps = []
val_rs = []
val_f1s = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(df_tokenized, df_tokenized['sentiment_label']):
    # 分割数据集
    train_df = df_tokenized.iloc[train_index]
    test_df = df_tokenized.iloc[test_index]

    train_dataset = CustomDataset(train_df, eye_tracking_features.iloc[train_index].values)
    test_dataset = CustomDataset(test_df, eye_tracking_features.iloc[test_index].values)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 定义模型
    model = get_model(model_type, num_eye_tracking_features, num_labels)
    model.to(device)
    # 定义优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)

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

    # 评估循环
    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(test_loader, desc=f"Validation", leave=True):
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
    val_accs.append(val_acc)
    val_ps.append(val_p)
    val_rs.append(val_r)
    val_f1s.append(val_f1)

# 输出平均的评估指标
print(f"Average Validation Acc: {np.mean(val_accs):.4f}")
print(f"Average Validation P: {np.mean(val_ps):.4f}")
print(f"Average Validation R: {np.mean(val_rs):.4f}")
print(f"Average Validation F1: {np.mean(val_f1s):.4f}")