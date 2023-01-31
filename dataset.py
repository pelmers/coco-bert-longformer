import torch
import pandas as pd

from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer, LongformerTokenizer, logging
from torch.nn.utils.rnn import pad_sequence
from constants import MAX_LEN

class CocoDataset(Dataset):
    def __init__(self, df, model_type):
        logging.set_verbosity_error()
        self.df = df
        self.model_type = model_type
        self.tokenizer = (
            BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
            if model_type == 'bert' else
            LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", do_lower_case=True)
        )
        self.max_len = min(512, MAX_LEN) if self.model_type == 'bert' else min(4096, MAX_LEN)
        self.data = self.load_data(self.df)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self, df):
        token_ids = []
        mask_ids = []
        seg_ids = []
        labels = []
        
        code_list = df['new_code_raw'].to_list() # for post hoc setting
        # code_list = df['span_diff_code_subtokens'].to_list() # for just-in-time setting
        comment_list = df['old_comment_raw'].to_list()
        label_list = df['label'].to_list()
        
        for (code, comment, label) in zip(code_list, comment_list, label_list):
            code_id = self.tokenizer.encode(code, add_special_tokens=False, truncation=True, max_length=self.max_len)
            comment_id = self.tokenizer.encode(comment, add_special_tokens=False, truncation=True, max_length=self.max_len)

            # want [CLS] comment tokens [SEP] code tokens [SEP]
            pair_token_ids = [self.tokenizer.cls_token_id] + comment_id + [self.tokenizer.sep_token_id] + code_id + [self.tokenizer.sep_token_id]
            pair_token_ids = self.truncate(pair_token_ids)
            code_len = len(code_id)
            comment_len = len(comment_id)
            
            attention_mask_ids = torch.tensor([1] * (code_len + comment_len + 3)) # mask padded values
            if self.model_type == 'longformer':
                segment_ids = torch.tensor([0] * (code_len + comment_len + 3)) # only for Longformer
            else:
                segment_ids = torch.tensor([0] * (comment_len + 2) + [1] * (code_len + 1)) # sentence 0 (comment) and sentence 1 (code)

            attention_mask_ids = self.truncate(attention_mask_ids)
            segment_ids = self.truncate(segment_ids)
            
            token_ids.append(torch.tensor(pair_token_ids))
            mask_ids.append(attention_mask_ids)
            seg_ids.append(segment_ids)
            labels.append(label)
            
        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        labels = torch.tensor(labels)
        
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, labels)
        return dataset

    def truncate(self, ids):
        return ids[:self.max_len] if len(ids) > self.max_len else ids

def retrieve_train_data(data_classes):
    return pd.concat([pd.read_json(f"./data/{data_class}/train.json") for data_class in data_classes], axis=0)

def retrieve_valid_data(data_classes):
    return pd.concat([pd.read_json(f"./data/{data_class}/valid.json") for data_class in data_classes], axis=0)

def retrieve_test_data(data_classes):
    return pd.concat([pd.read_json(f"./data/{data_class}/test.json") for data_class in data_classes], axis=0)