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

def get_broad_role(predicted_roles, probs, idx2label):
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

            return max_broad_role, max_role_contributors
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
        

        