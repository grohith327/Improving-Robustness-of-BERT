import sys

MASK_TOKEN = '[MASK]'
CLS_TOKEN = '[CLS]'

class ReplacementTracker:
    def __init__(self, f):
        lines = f.readlines()
        self.list = self.create_sentence_pairs(lines)
    def create_datafile(self):
        data = []
        for s_pair in self.list:
            mask_sentence = s_pair['orig']
            replacement_indices = s_pair['replacement_indices']
            for i, token in enumerate(mask_sentence):
                if i in replacement_indices:
                    mask_sentence[i] = MASK_TOKEN
            mask_sentence = ['[CLS]'] + mask_sentence
            data.append(mask_sentence)
        return data
    def create_sentence_pairs(self, lines):
        # pair_list: list of dictionaries, to be returned
        pair_list = []
        # sent_pair: dictionary that holds a pair of sentences
        # 'orig' for original and 'adv' for adversarial
        # and the list of replacements (a list of 2-element lists)
        sent_pair = {}
        sent_pair['replacements'] = []
        sent_pair['replacement_indices'] = []
        for line in lines:
            tokens = line.split()
            if tokens == []:
                continue
            key = tokens[0]               # either 'adv' or 'orig'
            assert tokens[1] == 'sent'
            label = tokens[2]             # either '(0):' or '(1):'
            label = int(label[1])         # extract 1 or 0 as int
            sent_pair[key] = tokens[3:]   # full sentence
            sent_pair[key + '_label'] = label
            # if both 'adv' and 'orig' are in the dict 'sent_pair', then
            # we figure out which words were replaced and store
            # them in sent_pair['replacements']
            if 'orig' in sent_pair.keys() and 'adv' in sent_pair.keys():
                # extract_replacements returns a list of 2-element lists
                # one 2-element list [original, replaced] for each replaced word
                sent_pair['replacements'] , sent_pair['replacement_indices'] = self.extract_replacements(
                    sent_pair['orig'],
                    sent_pair['adv'],
                )
                pair_list.append(sent_pair)
                sent_pair = {}            # prepare for next sentence pair
        return pair_list
    def extract_replacements(self, orig, adv):
        replacements = []
        indices = []
        for i in range(len(orig)):
            if orig[i] != adv[i]:
                replacements.append([orig[i], adv[i]])
                indices.append(i)
        return replacements, indices


