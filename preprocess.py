
import pandas as pd
import re

def extract_context_within_512(article, start, end, tokenizer, max_len=512):
    """
    Extract a context window around the entity mention, ensuring the total length 
    (entity + context) fits within max_len tokens, using correct mapping of character 
    offsets to token indices.
    """
    def add_entity_markers(article, start, end):
        """
        Add special markers around the entity in the text and adjust offsets.
        """
        before = article[:start]
        entity = article[start:end]
        after = article[end:]
        marked_article = before + "[ENTITY] " + entity + " [/ENTITY]" + after
        # 9 for len("[ENTITY] ")
        adjusted_start = start + 9
        adjusted_end = adjusted_start + len(entity)
        return marked_article, adjusted_start, adjusted_end
    
    marked_article, adjusted_start, adjusted_end = add_entity_markers(article, start, end)

    encoding = tokenizer(
        marked_article,
        return_offsets_mapping=True,
        truncation=False,
    )
    
    # Extract offset mapping (character to token mapping)
    offsets = encoding['offset_mapping']
    
    # Locate tokens corresponding to the entity (adjusted start and end)
    entity_token_indices = [
        idx for idx, (char_start, char_end) in enumerate(offsets)
        if char_start >= adjusted_start and char_end <= adjusted_end
    ]
    
    if not entity_token_indices:
        raise ValueError(f"Entity span does not match any token in the article: {marked_article}")
    
    # Get the first and last token indices of the entity (including markers)
    entity_start_token = entity_token_indices[0]
    entity_end_token = entity_token_indices[-1]
    
    # Determine the maximum context length available for tokens before and after the entity
    max_context_tokens = max_len - (entity_end_token - entity_start_token + 1) - 3 
    half_context_tokens = max_context_tokens // 2
    
    # Tokens before the entity
    tokens_before = encoding['input_ids'][
        max(0, entity_start_token - half_context_tokens):entity_start_token
    ]
    
    # Tokens after the entity
    tokens_after = encoding['input_ids'][
        entity_end_token + 1:min(len(encoding['input_ids']), entity_end_token + 1 + half_context_tokens)
    ]
    
    # Combine context and entity tokens
    context_tokens = tokens_before + encoding['input_ids'][entity_start_token:entity_end_token + 1] + tokens_after
    context = tokenizer.decode(context_tokens, skip_special_tokens=True)
    
    return context

# Add a context column to the DataFrame
def compute_context(row, articles, tokenizer):
    file_id = row['file_id']
    article = articles[file_id]
    start_offset = row['start_offset']
    end_offset = row['end_offset'] + 1
    return extract_context_within_512(
        article=article,
        start=start_offset,
        end=end_offset,
        tokenizer=tokenizer,
        max_len=512
    )


# Clean and split roles into proper lists
def clean_roles(roles):
    # Remove unwanted characters like '[' and ']' and extra quotes
    if isinstance(roles, str):
        roles = re.sub(r"[\[\]']", '', roles)  # Remove brackets and single quotes
        roles = [role.strip() for role in roles.split(',')]  # Split and strip whitespace
    return roles



# Convert fine-grained roles to multi-hot encoding
def encode_roles(roles, role_to_idx):
    labels = [0] * len(role_to_idx)
    for role in roles:
        labels[role_to_idx[role]] = 1
    return labels



