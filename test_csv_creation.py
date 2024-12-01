import pandas as pd

test_path = "/Users/mohamedelzeftawy/Documents/MBZUAI/NLP/Assignment2/dev-documents_25_October/EN/subtask-1-entity-mentions.txt"

lines = []
with open(test_path, 'r') as file:
    for line in file:
        # Split the line into columns
        parts = line.strip().split('\t')
        
        row = {
            "file_id": parts[0],
            "entity": parts[1],
            "start_offset": int(parts[2]),
            "end_offset": int(parts[3]),
        }
        lines.append(row)

test_df = pd.DataFrame(lines)
test_df.to_csv('test.csv', index=False)