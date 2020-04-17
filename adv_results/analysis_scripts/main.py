from track_replacements import ReplacementTracker
#from .mask_index import finetune_bert_on_custom_mlm

def main(filepath):
    with open(filepath, 'r') as f:
        reptracker = ReplacementTracker(f)
    for pair in reptracker.list:
        print(pair['orig'])
        print(pair['replacements'])
        print(pair['replacement_indices'])
    mlm_data = reptracker.create_datafile()
    print(mlm_data)
    with open('created_data/mlm_data.txt', 'w') as f:
        for line in mlm_data:
            joined_line = ' '.join(line)
            f.write(joined_line)
            f.write('\n')
    # finetune_bert_on_custom_mlm(mlm_data)

if __name__ == "__main__":
    main("../adversaries_ag.txt")
