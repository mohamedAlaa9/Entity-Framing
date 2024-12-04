from config import THRESHOLD
from sklearn.metrics import f1_score
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

def evaluate(model, dataloader, device, threshold=THRESHOLD):
    """
    Evaluate the model and compute F1 score.
    
    Args:
        dataloader: DataLoader for validation or test data.
        threshold: Probability threshold for multi-label classification.
    
    Returns:
        Micro F1 score.
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu()
            outputs = model(input_ids, attention_mask=attention_mask).cpu()

            loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
            total_loss += loss.item()
            labels = labels.numpy()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.append(labels)
            all_preds.append(probs)
    
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    binary_preds = (all_preds > threshold).astype(int)
    # print(all_labels)
    # print(binary_preds)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, f1_score(all_labels, binary_preds, average="macro")