import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm

from utils import get_broad_role

class MultiLabelClassifier:
    def __init__(self, idx_to_label, role_to_idx, model_name="bert-base-uncased", dropout_rate=0.3, device=None):
        """
        Initialize the MultiLabelClassifier with a pretrained model.
        
        Args:
            idx_to_label: Dictionary mapping label indices to names.
            role_to_idx: Dictionary mapping label names to indices.
            model_name: Pretrained model name or path.
            dropout_rate: Dropout rate for regularization.
            device: The device to use ("cuda" or "cpu").
        """
        self.idx_to_label = idx_to_label
        self.role_to_idx = role_to_idx
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(role_to_idx),
            id2label=idx_to_label,
            label2id=role_to_idx,
            problem_type="multi_label_classification"
        )
        # Adjust dropout
        self.model.config.hidden_dropout_prob = dropout_rate
        self.model.config.attention_probs_dropout_prob = dropout_rate
        
        # Freeze lower layers
        for param in self.model.bert.encoder.layer[:8].parameters():
            param.requires_grad = False
        
        self.model.to(self.device)

    def train(self, dataloader, optimizer, scheduler):
        """
        Train the model for one epoch with a progress bar.
        
        Args:
            dataloader: DataLoader for training data.
            optimizer: Optimizer for updating model weights.
            scheduler: Learning rate scheduler.
        
        Returns:
            Average loss over the epoch.
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training", unit="batch")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        progress_bar.close()
        return total_loss / len(dataloader)

    # def train(self, dataloader, optimizer, scheduler):
    #     """
    #     Train the model for one epoch.
        
    #     Args:
    #         dataloader: DataLoader for training data.
    #         optimizer: Optimizer for updating model weights.
    #         scheduler: Learning rate scheduler.
        
    #     Returns:
    #         Average loss over the epoch.
    #     """
    #     self.model.train()
    #     total_loss = 0
    #     for batch in dataloader:
    #         input_ids = batch['input_ids'].to(self.device)
    #         attention_mask = batch['attention_mask'].to(self.device)
    #         labels = batch['labels'].to(self.device)

    #         optimizer.zero_grad()
    #         outputs = self.model(input_ids, attention_mask=attention_mask)
    #         loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, labels)
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         total_loss += loss.item()
    #     return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, threshold=0.3):
        """
        Evaluate the model and compute F1 score.
        
        Args:
            dataloader: DataLoader for validation or test data.
            threshold: Probability threshold for multi-label classification.
        
        Returns:
            Micro F1 score.
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                
                all_labels.extend(labels)
                all_preds.extend(probs)
        
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)
        binary_preds = (all_preds > threshold).astype(int)
        return f1_score(all_labels, binary_preds, average="micro")
    
    def predict(self, dataloader, threshold=0.3):
        """
        Predict labels for the input data.
        
        Args:
            dataloader: DataLoader for prediction data.
            threshold: Probability threshold for multi-label classification.
        
        Returns:
            A tuple (true_labels, predictions).
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                
                all_labels.extend(labels)
                all_preds.extend(probs)
        
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)
        binary_preds = (all_preds > threshold).astype(int)
        return all_labels, binary_preds
    def predict_for_test_set_tab_format(self, tokenizer, test_data, output_file, idx_to_roles, threshold=0.31):
        """
        Predict labels for a test set and save the results in a tab-indented text format.
        
        Args:
            model: Trained PyTorch model.
            tokenizer: Tokenizer used during training.
            test_data: List of dictionaries with keys `file_id`, `text`, `start`, `end`.
                    Example: [{"file_id": "EN_UA_DEV_100012.txt", "text": "...", "start": 1441, "end": 1450}]
            device: The device (CPU/GPU) for computation.
            output_file: Path to save the prediction results.
            threshold: Probability threshold for multi-label predictions.
        """
        with open(output_file, "w") as f:
            for data in test_data:
                file_id = data["file_id"]
                text = data["context"]
                start = data["start_offset"]
                end = data["end_offset"]
                
                # Extract the entity from the text
                entity = data["entity"]
                
                # Predict roles for the text
                predicted_roles = self.predict_test(tokenizer, text, idx_to_roles, threshold)
                broad_role = get_broad_role(predicted_roles[0])

                # Convert roles to a comma-separated string
                roles_str = "\t".join(predicted_roles[0])
                
                # Write the result in the desired format
                f.write(f"{file_id}\t{entity}\t{start}\t{end}\t{broad_role}\t{roles_str}\n")
    def predict_test(self, tokenizer, text, idx_to_roles,  threshold=0.27):
        """
        Predict the roles for a given input text using a trained model.
        
        Args:
            model: The trained PyTorch model.
            tokenizer: The tokenizer used during training.
            text: The input text (string).
            device: The device (CPU/GPU) for computation.
            threshold: The probability threshold for binary classification.
            
        Returns:
            roles: A list of predicted roles for the input text.
        """
        self.model.eval()
        
        # Tokenize the input text
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move input data to the device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities
        # print(probs)
        # Apply threshold to convert probabilities to binary predictions
        binary_preds = (probs > threshold).astype(int)
        # Map indices to role names
        predicted_roles = []
        for indices in binary_preds:
            predicted_roles.append([idx_to_roles[idx] for idx, value in enumerate(indices) if value == 1])
        
        return predicted_roles