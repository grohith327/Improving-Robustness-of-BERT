from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

MAX_SEQ_LEN = 64
DATA_PATH = "../TextFooler/data/{}"
MODEL_PATH = "./BERTMLM_PRETRAINED"
EPOCHS = 5
BATCH_SIZE = 16
LR = 5e-5
EPSILON = 1e-8
MAX_GRAD_NORM = 1.0

device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def load_files():

    with open(DATA_PATH.format("mr"), "r") as f:
        data1 = f.read()
    data1 = data1.split("\n")

    with open(DATA_PATH.format("yelp"), "r") as f:
        data2 = f.read()
    data2 = data2.split("\n")

    with open(DATA_PATH.format("imdb"), "r") as f:
        data3 = f.read()
    data3 = data3.split("\n")

    return data1 + data2 + data3


def get_sent_labels():
    data = load_files()
    sentences = []
    labels = []
    for line in data:
        try:
            labels.append(int(line[0]))
            sentences.append(line[2:])
        except:
            pass
    return sentences, labels


def tokenize(sentences, data_type):
    input_tensors = []
    for sent in tqdm(sentences, desc=data_type):

        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=MAX_SEQ_LEN,
            pad_to_max_length=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )

        input_tensors.append(encoded_dict["input_ids"])

    return input_tensors


sentences, labels = get_sent_labels()
train_sent, test_sent, train_labels, test_labels = train_test_split(
    sentences, labels, stratify=labels, test_size=0.2, shuffle=True
)

train_tokens = tokenize(train_sent, "Train sentences")
test_tokens = tokenize(test_sent, "Test sentences")

train_tokens = torch.tensor(train_tokens)
test_tokens = torch.tensor(test_tokens)
train_labels = torch.tensor(train_labels, dtype=torch.float).unsqueeze(-1)
test_labels = torch.tensor(test_labels, dtype=torch.float).unsqueeze(-1)


class BertModel(nn.Module):
    def __init__(self, bert_path):
        super(BertModel, self).__init__()
        self.model = torch.load(bert_path)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(768 * 2, 2)

    def forward(self, input_tensors):

        _, hidden = self.model(input_tensors)
        hidden = hidden[-1]

        avg_pool = torch.mean(hidden, 1)
        max_pool, _ = torch.max(hidden, 1)

        output = torch.cat((avg_pool, max_pool), 1)
        output = self.dropout(output)
        output = self.linear(output)
        return output


model = BertModel(MODEL_PATH)
model.to(device)

train_data = TensorDataset(train_tokens, train_labels)
test_data = TensorDataset(test_tokens, test_labels)

train_dataloader = DataLoader(
    train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE
)
test_dataloader = DataLoader(
    test_data, sampler=RandomSampler(test_data), batch_size=BATCH_SIZE
)

optimizer = AdamW(model.parameters(), lr=LR, eps=EPSILON)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
criterion = nn.BCELoss()

model.zero_grad()
train_total = train_labels.size()[0]
test_total = test_labels.size()[0]
for epoch in range(EPOCHS):
    pbar = tqdm(total=train_total // BATCH_SIZE, desc="Epoch - " + str(epoch + 1))
    train_loss = 0.0
    test_loss = 0.0
    train_correct = 0
    test_correct = 0
    for batch_tokens, batch_labels in train_dataloader:

        batch_tokens = batch_tokens.to(device)
        batch_labels = batch_labels.to(device)

        model.train()
        preds = model(batch_tokens)
        preds = nn.functional.softmax(preds, dim=1)
        y_onehot = torch.nn.functional.one_hot(
            batch_labels.squeeze(-1).type(torch.long)
        ).type(torch.float)
        loss = criterion(preds, y_onehot)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        train_loss += loss.item()
        pred_labels = torch.argmax(preds, dim=1)
        train_correct += (pred_labels == batch_labels.squeeze(-1)).sum().item()
        pbar.update(1)

    pbar.close()
    del pbar

    model.eval()
    with torch.no_grad():
        for batch_tokens, batch_labels in test_dataloader:

            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)

            preds = model(batch_tokens)
            preds = nn.functional.softmax(preds, dim=1)
            y_onehot = torch.nn.functional.one_hot(
                batch_labels.squeeze(-1).type(torch.long)
            ).type(torch.float)
            loss = criterion(preds, y_onehot)
            test_loss += loss.item()

            pred_labels = torch.argmax(preds, dim=1)
            test_correct += (pred_labels == batch_labels.squeeze(-1)).sum().item()

    print(
        "Epoch:{} loss:{} accuracy:{} test loss:{} test accuracy:{}".format(
            epoch + 1,
            train_loss / float(train_total),
            train_correct / train_total,
            test_loss / float(test_total),
            test_correct / test_total,
        )
    )

torch.save(model.state_dict(), "./BertSentimentClassfication")
print("Model Saved")
