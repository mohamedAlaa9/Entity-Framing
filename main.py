import torch
from torch.utils.data import DataLoader
import config
from dataset import EntityDataset
from model import MultiLabelClassifier
from utils import set_seed, save_model, load_model
import pandas as pd
from config import TRAIN_PATH, TEST_PATH, OUTPUT_PATH
from transformers import AutoTokenizer, DebertaV2Tokenizer
from preprocess import compute_context, clean_roles, encode_roles
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from train import train
from eval import evaluate
from predict import predict_for_test_set_tab_format
import argparse

def main(num, model_path):
    if model_path is not None:
        
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)        
        train_data = pd.read_csv(TRAIN_PATH)
        train_data['fine_grained_roles'] = train_data['fine_grained_roles'].apply(clean_roles)
        # Extract unique roles
        all_roles = set(role for roles in train_data['fine_grained_roles'] for role in roles)
        role_to_idx = {role: idx for idx, role in enumerate(sorted(all_roles))}
        idx_to_roles = {idx: role for role, idx in role_to_idx.items()}
        device = torch.device(config.DEVICE)
        model = load_model(MultiLabelClassifier,idx_to_roles, role_to_idx,  model_path, device)
        test_articles = {}
        test_data = pd.read_csv(TEST_PATH)
        for file_id in test_data['file_id'].unique():
            with open(f"dev-documents_25_October/EN/subtask-1-documents/{file_id}", "r") as f:
                test_articles[file_id] = f.read()
        test_data['context'] = test_data.apply(compute_context, axis=1, args=(test_articles, tokenizer))
        path = OUTPUT_PATH + f"{num}"
        # mentions = test_data['entity'].tolist()
        pred_path = path + "/prediction.txt"
        predict_for_test_set_tab_format(model, tokenizer, test_data.to_dict(orient='records'), pred_path, idx_to_roles, role_to_idx, device,  threshold=config.THRESHOLD)
        return
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)

    device = torch.device(config.DEVICE)
     # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)    
    # Load data
    train_data = pd.read_csv(TRAIN_PATH)
    articles = {}
    for file_id in train_data['file_id'].unique():
        with open(f"training_data_16_October_release/EN/raw-documents/{file_id}", "r") as f:
            articles[file_id] = f.read()
    train_data['context'] = train_data.apply(compute_context, axis=1, args=(articles, tokenizer))
    # Apply cleaning to the column
    train_data['fine_grained_roles'] = train_data['fine_grained_roles'].apply(clean_roles)
    # Extract unique roles
    all_roles = set(role for roles in train_data['fine_grained_roles'] for role in roles)
    role_to_idx = {role: idx for idx, role in enumerate(sorted(all_roles))}
    idx_to_roles = {idx: role for role, idx in role_to_idx.items()}

    train_data['labels'] = train_data['fine_grained_roles'].apply(encode_roles, args=(role_to_idx,))
    mentions = train_data['entity'].tolist()
    contexts = train_data['context'].tolist()
    labels = train_data['labels'].tolist()

   

    # Split the data into training and validation sets
    mentions_train, mentions_val, contexts_train, contexts_val, labels_train, labels_val = train_test_split(
        mentions, contexts, labels, test_size=0.1, random_state=42
    )

    # Create the training and validMODEL_NAMEation datasets
    train_dataset = EntityDataset(mentions_train, contexts_train, labels_train, tokenizer, max_len=512)
    val_dataset = EntityDataset(mentions_val, contexts_val, labels_val, tokenizer, max_len=512)

    # Create the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)    
    print("Before model")

    model = MultiLabelClassifier(
        idx_to_label=idx_to_roles,
        role_to_idx=role_to_idx,
        model_name=config.MODEL_NAME,
    ).to(device)
    print("After Model")
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)
    best_val_loss = float("inf")
    path = OUTPUT_PATH + f"{num}"

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train the model
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate the model
        val_loss, val_f1 = evaluate(model, val_dataloader, device, threshold=config.THRESHOLD)
        print(f"Validation F1: {val_f1:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, path)
            print(f"Saved model with validation loss: {val_loss:.4f} at epoch {epoch+1}")

    test_articles = {}
    test_data = pd.read_csv(TEST_PATH)
    for file_id in test_data['file_id'].unique():
        with open(f"dev-documents_25_October/EN/subtask-1-documents/{file_id}", "r") as f:
            test_articles[file_id] = f.read()
    test_data['context'] = test_data.apply(compute_context, axis=1, args=(test_articles, tokenizer))
    pred_path = path + "/prediction.txt"
    predict_for_test_set_tab_format(model, tokenizer, test_data.to_dict(orient='records'), pred_path, idx_to_roles, role_to_idx, device,  threshold=config.THRESHOLD)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True, help="Experiment number for tracking and saving results.")
    parser.add_argument("--model_path", type=str, required=False, default=None, help="Path to the model to be loaded.")

    args = parser.parse_args()
    main(args.num, args.model_path)