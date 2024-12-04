import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from config import THRESHOLD, DROPOUT_RATE
import torch.nn as nn
from transformers import BertModel, AutoModel
from utils import get_broad_role


class MultiLabelClassifier(nn.Module):
    def __init__(self, idx_to_label, role_to_idx, model_name="bert-base-uncased", dropout_rate=DROPOUT_RATE):
        super(MultiLabelClassifier, self).__init__()
        cache_dir = "./hf_cache"
        self.bert = AutoModel.from_pretrained(model_name, cache_dir = cache_dir)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(role_to_idx))
        self.idx_to_label = idx_to_label
        self.role_to_idx = role_to_idx
        # torch.cuda.empty_cache() 

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.bert(input_ids, attention_mask=attention_mask)
    #     pooled_output = outputs.pooler_output
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #     return logits
        
    # Adjust to use deberta
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Use the last hidden state (first token [CLS] embedding) for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

