import torch
import random
import numpy as np
import os
import config
protagonist_roles = [
    "Guardian",
    "Martyr",
    "Peacemaker",
    "Rebel",
    "Underdog",
    "Virtuous"
]
antagonist_roles = [
    "Instigator",
    "Conspirator",
    "Tyrant",
    "Foreign Adversary",
    "Traitor",
    "Spy",
    "Saboteur",
    "Corrupt",
    "Incompetent",
    "Terrorist",
    "Deceiver",
    "Bigot"
]
innocent_roles = [
    "Forgotten",
    "Exploited",
    "Victim",
    "Scapegoat"
]

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, path):
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    path = path + '/model_state.pth'
    torch.save(model.state_dict(), path)

def load_model(model_class, idx_to_roles, role_to_idx, path, device="cpu"):
    """
    Loads a PyTorch model saved as a state dictionary.

    Args:
        model_class (torch.nn.Module): The class of the model to be loaded.
        path (str): The directory path where the model state file is saved.
        device (str): Device to load the model onto (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    model_state_path = os.path.join(path, "model_state.pth")
    
    if not os.path.exists(model_state_path):
        raise FileNotFoundError(f"Model state file not found at {model_state_path}")
    
    # Initialize the model architecture
    model = model_class(idx_to_label=idx_to_roles,
        role_to_idx=role_to_idx,
        model_name = config.MODEL_NAME)
    
    # Load the saved state dictionary into the model
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    
    # Move the model to the specified device
    model.to(device)
    
    return model

def get_broad_role(predicted_roles, probs, idx2label, label2idx):
        """
        Determine the broad role based on the majority of fine-grained roles.
        """
        counts = {
            "Protagonist": sum(1 for role in predicted_roles if role in protagonist_roles),
            "Antagonist": sum(1 for role in predicted_roles if role in antagonist_roles),
            "Innocent": sum(1 for role in predicted_roles if role in innocent_roles),
        }
        # Determine the broad role with the highest count
        if max(counts.values()) > 0:
            max_broad_role = max(counts, key=counts.get)
            # Get the predicted roles that contributed to the broad role
            max_role_contributors = []
            if max_broad_role == "Protagonist":
                max_role_contributors = [role for role in predicted_roles if role in protagonist_roles]
            elif max_broad_role == "Antagonist":
                max_role_contributors = [role for role in predicted_roles if role in antagonist_roles]
            elif max_broad_role == "Innocent":
                max_role_contributors = [role for role in predicted_roles if role in innocent_roles]
            
            # Get indices of max_role_contributors
            max_role_indices = [label2idx[role] for role in max_role_contributors]
            # Get top 3 contributors based on probs
            top_3_indices = sorted(max_role_indices, key=lambda x: probs[x], reverse=True)[:5]

            # Randomly select one from the top 3
            random_index = random.choice(top_3_indices)

            # Convert the selected index back to the role
            selected_role = idx2label[random_index]
            max_contributors = []
            max_contributors.append(selected_role)
            return max_broad_role, max_contributors
            # # Get top 2 contributors based on probs
            # top_2_indices = sorted(max_role_indices, key=lambda x: probs[x], reverse=True)[:2]

            # # Convert indices back to roles
            # top_2_roles = [idx2label[idx] for idx in top_2_indices]
            # return max_broad_role, top_2_roles
        else:

            probs = list(probs)
            # Use the max probability to determine the broad role when counts are zero => nothing is predicted
            max_prob_index = probs.index(max(probs))
            max_contributors = []
            max_prob_role = idx2label[max_prob_index]
            max_contributors.append(max_prob_role)
            if max_prob_role in protagonist_roles:
                max_broad_role = "Protagonist"
            elif max_prob_role in antagonist_roles:
                max_broad_role = "Antagonist"
            elif max_prob_role in innocent_roles:
                max_broad_role = "Innocent"
            return max_broad_role, max_contributors
        

        