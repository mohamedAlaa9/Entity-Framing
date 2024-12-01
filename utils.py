import torch
import random
import numpy as np
import os
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

def save_model(model, tokenizer, path):
        # Save the trained model and tokenizer

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Save model
    model.save_pretrained(path, safe_serialization=False)
    # Save tokenizer
    tokenizer.save_pretrained(path)

def get_broad_role(predicted_roles):
        """
        Determine the broad role based on the majority of fine-grained roles.
        """
        counts = {
            "Protagonist": sum(1 for role in predicted_roles if role in protagonist_roles),
            "Antagonist": sum(1 for role in predicted_roles if role in antagonist_roles),
            "Innocent": sum(1 for role in predicted_roles if role in innocent_roles),
        }
        # Return the broad role with the highest count
        return max(counts, key=counts.get) if max(counts.values()) > 0 else "Unknown"
