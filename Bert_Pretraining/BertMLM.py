from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm, trange
import glob

## Params
BATCH_SIZE = 16
MAX_SEQ_LEN = 128
LR = 5e-5
EPSILON = 1e-8
EPOCHS = 250
MAX_GRAD_NORM = 1.0
dataset = "imdb"
SAVE_PATH = "./BERT_MLM_{}.bin".format(dataset)
DATA_PATH = "./adv_results/*/*"

device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def load_files():

    files = glob.glob(DATA_PATH)
    files = list(filter(lambda x: dataset in x and ".txt" in x, files))

    data = None
    for i, file in enumerate(files):
        with open(file, "r") as f:
            lines = f.read()
        lines = lines.split("\n")
        if i == 0:
            data = lines
        else:
            data += lines

    return data


data = load_files()
print("Data Loaded, Data Length:{}".format(len(data)))

org_sent = []
adv_sent = []
for line in data:
    if len(line) == 0:
        continue
    txt_type, line = line.split("\t")[0], line.split("\t")[1]
    if txt_type[:4] == "orig":
        org_sent.append(line.lower())
    else:
        adv_sent.append(line.lower())

masks = []
for org_line, adv_line in zip(org_sent, adv_sent):
    org_line = org_line.split(" ")
    adv_line = adv_line.split(" ")
    mask_pos = []
    i = 0
    for org_word, adv_word in zip(org_line, adv_line):
        if org_word != adv_word:
            mask_pos.append(i)
        i += 1
    masks.append(mask_pos)

input_tensors = []
labels = []
for i in tqdm(range(len(adv_sent)), total=len(adv_sent)):
    masked_sent = adv_sent[i].split(" ")
    for pos in masks[i]:
        masked_sent[pos] = "[MASK]"
    masked_sent = " ".join(masked_sent)

    encoded_dict = tokenizer.encode_plus(
        masked_sent,
        add_special_tokens=True,
        max_length=MAX_SEQ_LEN,
        pad_to_max_length=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_special_tokens_mask=False,
    )
    input_tensors.append(encoded_dict["input_ids"])

    encoded_dict = tokenizer.encode_plus(
        adv_sent[i],
        add_special_tokens=True,
        max_length=MAX_SEQ_LEN,
        pad_to_max_length=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_special_tokens_mask=False,
    )
    labels.append(encoded_dict["input_ids"])

input_tensors = torch.tensor(input_tensors)
labels = torch.tensor(labels)

model = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
model.to(device)

train_data = TensorDataset(input_tensors, labels)
train_dataloader = DataLoader(
    train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE
)

optimizer = AdamW(model.parameters(), lr=LR, eps=EPSILON)

total_steps = len(train_dataloader) / BATCH_SIZE * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

tot_loss = 0.0
steps = 0.0

model.zero_grad()

for epoch in range(EPOCHS):

    for step, batch in enumerate(train_dataloader):

        batch_input_tensors = batch[0].to(device)
        batch_labels = batch[1].to(device)

        model.train()
        outputs = model(batch_input_tensors, masked_lm_labels=batch_labels)
        loss = outputs[0]

        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        steps += 1.0

    print("Epoch:{} loss:{}".format(epoch + 1, tot_loss / steps))

torch.save(model.state_dict(), SAVE_PATH)
print("Model saved")
