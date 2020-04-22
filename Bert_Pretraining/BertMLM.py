from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset 
from tqdm import tqdm, trange

## Params
BATCH_SIZE = 32
MAX_SEQ_LEN = 64
LR = 5e-5
EPSILON = 1e-8
EPOCHS = 150
MAX_GRAD_NORM = 1.0
SAVE_PATH = './BERT_PRETRAINED'
DATA_PATH = './adv_data/{}/adversaries.txt'

device = None
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_files():

    with open(DATA_PATH.format('mr_results'), 'r') as f:
        data1 = f.read()
    data1 = data1.split('\n')

    with open(DATA_PATH.format('yelp_results/yelp_part1'), 'r') as f:
        data2 = f.read()
    data2 = data2.split('\n')

    with open(DATA_PATH.format('yelp_results/yelp_part2'), 'r') as f:
        data3 = f.read()
    data3 = data3.split('\n')

    with open(DATA_PATH.format('imdb_results'), 'r') as f:
        data4 = f.read()
    data4 = data4.split('\n')

    return data1 + data2 + data3 + data4

data = load_files()

org_sent = []
adv_sent = []
for line in data:
    if(len(line) == 0):
        continue
    txt_type, line = line.split('\t')[0], line.split('\t')[1]
    if(txt_type[:4] == 'orig'):
        org_sent.append(line.lower())
    else:
        adv_sent.append(line.lower())

masks = []
for org_line, adv_line in zip(org_sent, adv_sent):
    org_line = org_line.split(' ')
    adv_line = adv_line.split(' ')
    mask_pos= []
    i = 0
    for org_word, adv_word in zip(org_line, adv_line):
        if(org_word != adv_word):
            mask_pos.append(i)
        i += 1
    masks.append(mask_pos)

input_tensors = []
labels = []
for i in tqdm(range(len(adv_sent)), total = 717):
    masked_sent = adv_sent[i].split(' ')
    for pos in masks[i]:
        masked_sent[pos] = '[MASK]'
    masked_sent = ' '.join(masked_sent)

    encoded_dict = tokenizer.encode_plus(masked_sent,
                                         add_special_tokens = True,
                                         max_length = MAX_SEQ_LEN,
                                         pad_to_max_length = True,
                                         return_token_type_ids = False,
                                         return_attention_mask = False,
                                         return_special_tokens_mask = False)                
    input_tensors.append(encoded_dict['input_ids'])

    encoded_dict = tokenizer.encode_plus(adv_sent[i],
                                         add_special_tokens = True,
                                         max_length = MAX_SEQ_LEN,
                                         pad_to_max_length = True,
                                         return_token_type_ids = False,
                                         return_attention_mask = False,
                                         return_special_tokens_mask = False)
    labels.append(encoded_dict['input_ids'])                                         

input_tensors = torch.tensor(input_tensors)
labels = torch.tensor(labels)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(device)

train_data = TensorDataset(input_tensors, labels)
train_dataloader = DataLoader(train_data, sampler = RandomSampler(train_data), batch_size = BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr = LR, eps = EPSILON)

total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

tot_loss = 0.0
steps = 0.0

model.zero_grad()

for epoch in range(EPOCHS):

    for step, batch in enumerate(train_dataloader):

        batch_input_tensors = batch[0].to(device)
        batch_labels = batch[1].to(device)

        model.train()
        outputs = model(batch_input_tensors, masked_lm_labels = batch_labels)
        loss = outputs[0]

        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        steps += 1.0

    print('Epoch:{} loss:{}'.format(epoch+1, tot_loss/steps))

torch.save(model, SAVE_PATH)
print('Model saved')