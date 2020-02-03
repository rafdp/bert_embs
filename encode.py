
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import numpy as np
from bert.tokenization import FullTokenizer
import time
import re
import sys
import json
hparams = json.load(open(sys.argv[1]))

MIN_CHAR_LEN = hparams['min_char_len']
MAX_CHAR_LEN = hparams['max_char_len']
MAX_SEQ_LEN  = hparams['max_seq_len']

class PreBertEncoder(object):
    def __init__(self, dataset_path = 'train.csv'):
        self.dataset = pd.read_csv(dataset_path)
        for _, row in self.dataset.iterrows():
            if row['target'] > 0.8:
                print("==================================================================")
                print("==================================================================")
                print("==================================================================")
                print(row['comment_text'])
    @staticmethod
    def CalcUppercasePercentage(st):
        count = 0
        for ch in st:
            if ch.isupper():
                count += 1
        return count*1.0/len(st)

    def PrimaryFiltering(self, min_char_len = MIN_CHAR_LEN, 
                         max_char_len = MAX_CHAR_LEN, 
                         max_seq_len = MAX_SEQ_LEN, 
                         cut_freq = None, 
                         cut_idx = None):
        self.max_seq_len = max_seq_len
        len_filter_mask = (self.dataset['comment_text'].str.len() >= min_char_len) & \
                           (self.dataset['comment_text'].str.len() < max_char_len)
        nbefore = len(self.dataset)
        self.dataset = self.dataset.loc[len_filter_mask]
        self.dataset = self.dataset[self.dataset["psychiatric_or_mental_illness"].notna()]
        if cut_freq != None:
            self.dataset = self.dataset.loc[cut_idx::cut_freq]
        self.dataset.reset_index(drop=True, inplace=True)
        self.input_uppercase_coeff = []
        for i in range(len(self.dataset)):
            self.input_uppercase_coeff.append(
                 self.CalcUppercasePercentage(self.dataset["comment_text"][i])
                                             )
        self.input_uppercase_coeff = np.array(self.input_uppercase_coeff)
        filter_lambda = lambda st: np.char.replace((' '.join(
                                     st.split()[0:max_seq_len]
                                                            )
                                                   ).lower(), '\\n', ' ')
                                        
        self.dataset['comment_text'] = self.dataset['comment_text'].apply(filter_lambda)
        nafter = len(self.dataset)
        print("Removed %.2f %% of dataset ( filtering len [%d, %d) chars, truncating at %d words)" % 
              ((nbefore - nafter)/(1.0*nbefore)*100, 
              min_char_len, max_char_len, 
              max_seq_len))
        print("New size %d" % len(self.dataset))

    def BuildIdentityData(self, columns = hparams['identity_columns']):

        self.identity_data = (np.nan_to_num(
                               self.dataset[columns].to_numpy()
                              ) >= 0.5).astype(int)
        self.identity_weights = []
        for i in range(np.shape(self.identity_data)[1]):
            self.identity_weights.append(np.sum(self.identity_data[:, i]/len(self.identity_data)))
        self.identity_weights = np.array(self.identity_weights)

    def BuildTargetData(self, columns = hparams['target_columns']):
        self.target_data = (np.nan_to_num(
                             self.dataset[columns].to_numpy()
                                    ) >= 0.5).astype(int)
        self.target_weights = []
        for i in range(np.shape(self.target_data)[1]):
            self.target_weights.append(np.sum(self.target_data[:, i]/len(self.target_data)))
        self.target_weights = np.array(self.target_weights)

    def BuildInputData(self, column = 'comment_text'):
        self.input_data = np.array(self.dataset[column].to_numpy())

    def LazyBuildData(self):
        self.BuildIdentityData()
        self.BuildTargetData()
        self.BuildInputData()
    
    def InitializeBertTokenizer(self, 
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

        sess = tf.Session()
        bert_module =  hub.Module(bert_path, trainable = False)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    
        vocab_file, do_lower_case = sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"],
            ]
        )
    
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    
    @staticmethod
    def ProcessSingleComment_FirstPass(comment, max_seq_len, tokenizer):
        tokens = ["[CLS]"]
        tokens.extend(tokenizer.tokenize(re.sub('\s{2,}', ' ', re.sub('([^a-zA-Z])', r' \1 ', comment))))
        if len(tokens) > (max_seq_len - 1):
            tokens = tokens[:max_seq_len - 1]
        tokens.append("[SEP]")
        return tokens, len(tokens)
    
    @staticmethod
    def ProcessSingleComment_SecondPass(tokens, max_seq_len, tokenizer):
        input_id = np.pad(np.array(tokenizer.convert_tokens_to_ids(tokens[0])), 
                             (0, max_seq_len-tokens[1]), "constant", constant_values=(0, 0))
        segment_id = np.zeros(max_seq_len)
        input_mask = np.zeros(max_seq_len)
        input_mask[:tokens[1]+1] = 1
        return input_id, segment_id, input_mask
    
    def ProcessDataset(self):
        data_1 = []
        max_seq_len = 0
        start_time = time.process_time()
        last_checkpoint = start_time
        for i in range(len(self.dataset)):
            if (i % 100 == 1) and time.process_time() - last_checkpoint > 10.0:
                print("Pass 1: %d%%, estimated time remaining %d s    \r" % 
                      (i / (len(self.input_data)*1.0)*100, 
                       (time.process_time()-start_time)/i*(len(self.input_data)-i)), 
                     end="")
                last_checkpoint = time.process_time()
            tokens = self.ProcessSingleComment_FirstPass(self.input_data[i], self.max_seq_len, self.tokenizer)
            if tokens[1] > max_seq_len:
                max_seq_len = tokens[1]
            data_1.append(tokens)
        print("")
        self.input_ids = []
        self.segment_ids = []
        self.input_masks = []
        start_time = time.process_time()
        last_checkpoint = start_time
        for i in range(len(self.input_data)):
            if (i % 100 == 1) and time.process_time() - last_checkpoint > 10.0:
                print("Pass 2: %d%%, estimated time remaining %d s    \r" % 
                      (i / (len(self.input_data)*1.0)*100, 
                       (time.process_time()-start_time)/i*(len(self.input_data)-i)), 
                     end="")
                last_checkpoint = time.process_time()
            input_id, segment_id, input_mask = self.ProcessSingleComment_SecondPass(data_1[i], self.max_seq_len, self.tokenizer)
            self.input_ids.append(input_id)
            self.segment_ids.append(segment_id)
            self.input_masks.append(input_mask)

    def Save(self, path='encoded_data'): 
        next_stage_info = np.array([{"target_weights": self.target_weights,
                                     "identity_weights": self.identity_weights,
                                     "uppercase_coeff": self.input_uppercase_coeff,
                                     "identity_data": self.identity_data,
                                     "target_data": self.target_data,
                                     "bert_input_data": {"input_ids": self.input_ids,
                                                         "input_masks": self.input_masks,
                                                         "segment_ids": self.segment_ids}}])
        print("Saving info to '%s'" % path)  
        np.save(path, next_stage_info)

if __name__ == "__main__":
    enc = PreBertEncoder()
    enc.PrimaryFiltering()
    enc.LazyBuildData()
    enc.InitializeBertTokenizer()
    enc.ProcessDataset()
    enc.Save(hparams['data_path'])
