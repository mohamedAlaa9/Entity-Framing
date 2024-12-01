import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
class EntityDataset(Dataset):
    def __init__(self, mentions, contexts, labels, tokenizer, max_len):
        self.mentions = mentions
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        mention = self.mentions[idx]
        context = self.contexts[idx]
        label = self.labels[idx]


        inputs = self.tokenizer(
            context,
            truncation=True, max_length=self.max_len, 
            padding="max_length", return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }