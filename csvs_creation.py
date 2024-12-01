import pandas as pd

# Path to your annotations file
annotations_files = ["/Users/mohamedelzeftawy/Documents/MBZUAI/NLP/Assignment2/training_data_16_October_release/EN/subtask-1-annotations.txt",
                      "/Users/mohamedelzeftawy/Documents/MBZUAI/NLP/Assignment2/training_data_16_October_release/BG/subtask-1-annotations.txt",
                      "/Users/mohamedelzeftawy/Documents/MBZUAI/NLP/Assignment2/training_data_16_October_release/HI/subtask-1-annotations.txt",
                      "/Users/mohamedelzeftawy/Documents/MBZUAI/NLP/Assignment2/training_data_16_October_release/PT/subtask-1-annotations.txt"]
for annotations_file in annotations_files:
    # Read the file
    annotations = []
    with open(annotations_file, 'r') as file:
        for line in file:
            # Split the line into columns
            parts = line.strip().split('\t')
            
            # Store the parsed line as a dictionary
            annotation = {
                "file_id": parts[0],
                "entity": parts[1],
                "start_offset": int(parts[2]),
                "end_offset": int(parts[3]),
                "broad_roles": parts[4] if len(parts) > 4 else [],
                "fine_grained_roles": parts[5:] if len(parts) > 5 else []
            }
            # if len(parts) > 6:
            #     print(parts[5]) if parts[5] == "Antagonist" or parts[5] == "Protagonist" or parts[5] == "Innocent" else print("Not Found")
            annotations.append(annotation)

    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv(annotations_file.split('/')[8].split('.')[0] + '_annotations.csv', index=False)