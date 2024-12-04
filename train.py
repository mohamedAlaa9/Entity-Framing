import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, scheduler, device):
    """
    Train the model for one epoch with a progress bar.

    Args:
        dataloader: DataLoader for training data.
        optimizer: Optimizer for updating model weights.
        scheduler: Learning rate scheduler.

    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    progress_bar.close()
    return total_loss / len(dataloader)