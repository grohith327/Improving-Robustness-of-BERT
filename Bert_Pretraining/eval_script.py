import torch
from transformers import BertTokenizer
from tqdm import tqdm  

DATA_PATH = './adv_data/{}/adversaries.txt'
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(sent, dtype):
    input_tensors = []
    attn_masks = []
    for line in tqdm(sent, desc = dtype):
        encoded_dict = tokenizer.encode_plus(line,
                                            add_special_tokens = True, 
                                            max_length = 128,
                                            pad_to_max_length = True,
                                            return_attention_mask = True)
        input_tensors.append(encoded_dict['input_ids'])
        attn_masks.append(encoded_dict['attention_mask'])
    
    return input_tensors, attn_masks

adv_tensors, adv_attn_masks = tokenize(adv_sent, 'Adversarial Sentences')
org_tensors, org_attn_masks = tokenize(org_sent, 'Original Sentences')

adv_tensors = torch.tensor(adv_tensors)
adv_tensors = adv_tensors.to('cuda')

model = torch.load('./BERT_PRETRAINED')
model.to('cuda')
model.eval()
print(model)

output = model(adv_tensors[0].unsqueeze(0))