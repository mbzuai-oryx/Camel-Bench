import json
import pandas as pd
import random
from utils import medicalmmt_eval

def filter_data_perct(path, results_path, remove_percent):
    df = pd.read_parquet(path)
    with open(results_path, "r") as f:
        results = json.load(f)
    
    correct_indices = [i['index'] for i in results if medicalmmt_eval(i['pred_answer'], i['answer']) and i['index'] < len(df)]
    
    total_samples = len(df)
    correct_samples = len(correct_indices)
    current_accuracy = correct_samples / total_samples
    
    target_accuracy = max(0, current_accuracy - (remove_percent / 100))
    samples_to_remove = int(correct_samples - (target_accuracy * total_samples))
    indices_to_remove = random.sample(correct_indices, samples_to_remove)
    df = df.drop(indices_to_remove)
    df = df.reset_index(drop=True)
    
    return df

def filter_data(path, results_path):
    df = pd.read_parquet(path)
    with open(results_path, "r") as f:
        results = json.load(f)

    indices = [i['index'] for i in results]

    df = df.drop(indices)
    df = df.reset_index(drop=True)
    return df

def filter_none_images(path):
    df = pd.read_parquet(path)
    df = df[df['image'].notna()]
    df = df.reset_index(drop=True)
    return df


remove_percent = 20
# path = "medical-mmt-mi_checked.parquet"
# results_path = "results_merged.json"
# filtered_df = filter_data(path, results_path)
# filtered_df.to_parquet("data_prunedv1.parquet", index=False)

path = "data_prunedv1.parquet"
results_path = "results_gpt.json"
filtered_df = filter_data_perct(path, results_path, remove_percent)
filtered_df.to_parquet("data_prunedv2.parquet", index=False)

# path = "data_prunedv1.parquet"
# filtered_df = filter_none_images(path)
# filtered_df.to_parquet("data_prunedv2.parquet", index=False)

print(f"Original dataframe shape: {pd.read_parquet(path).shape}")
print(f"Filtered dataframe shape: {filtered_df.shape}")
print(f"Removed {remove_percent}% of correct samples")