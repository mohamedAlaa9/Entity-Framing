from config import THRESHOLD
from utils import get_broad_role
import torch

def predict_for_test_set_tab_format(model, tokenizer, test_data, output_file, idx_to_roles, roles_to_idx, device, threshold=THRESHOLD):
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
            
            # Extraselfct the entity from the text
            entity = data["entity"]
            
            # Predict roles for the text
            predicted_roles, probs = predict_test(model, tokenizer, text, entity, idx_to_roles, device, threshold)
            broad_role, fine_grained_roles = get_broad_role(predicted_roles[0], probs, idx_to_roles, roles_to_idx)

            # Convert roles to a comma-separated string
            roles_str = "\t".join(fine_grained_roles)
            
            # Write the result in the desired format
            f.write(f"{file_id}\t{entity}\t{start}\t{end}\t{broad_role}\t{roles_str}\n")
def predict_test(model, tokenizer, text, entity, idx_to_roles, device ,threshold=THRESHOLD):
    """
    Predict the roles for a give model_name="bert-base-uncased",
    dropout_rate=0.3n input text using a trained model.
    
    Args:
        model: The trained PyTorch model.
        tokenizer: The tokenizer used during training.
        text: The input text (string).
        device: The device (CPU/GPU) for computation.
        threshold: The probability threshold for binary classification.
        
    Returns:
        roles: A list of predicted roles for the input text.
    """
    model.eval()
    combined_text = f"{entity} [SEP] {text}"
    inputs = tokenizer(
        combined_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs
        probs = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities
    # print(probs)
    # Apply threshold to convert probabilities to binary predictions
    binary_preds = (probs > threshold).astype(int)
    # Map indices to role names
    predicted_roles = []
    for indices in binary_preds:
        predicted_roles.append([idx_to_roles[idx] for idx, value in enumerate(indices) if value == 1])
    print(probs)
    return predicted_roles, probs[0]