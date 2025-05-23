import os

# File paths
# Define the subfolder path
subfolder = os.path.join(os.path.dirname(__file__), "data", "CHEMINFORMATICS")
input_filename = "reactionSmilesFigShareUSPTO2023.txt"
input_filepath = os.path.join(subfolder, input_filename)

# Output filename in the same subfolder
base, ext = os.path.splitext(input_filename)
output_filename = f"{base}_filtered{ext}"
output_filepath = os.path.join(subfolder, output_filename)

CONTEXT_SIZE = 256
max_length = CONTEXT_SIZE - 1



with open(input_filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8') as outfile:
    for line in infile:
        formula = line.rstrip('\n')
        if len(formula) <= max_length:
            outfile.write(line)

print(f"Filtering complete. Filtered data saved to: {output_filepath}")

# --- Split filtered data into train/val/test sets (85/10/5) ---
import random

# Read all filtered lines
with open(output_filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.shuffle(lines)

total = len(lines)
n_train = int(total * 0.85)
n_val = int(total * 0.10)
# Ensure all lines are used (test gets the remainder)
n_test = total - n_train - n_val

# Toggle: Set to True to perform real splits, False to put all data in 'trn'.
do_real_split = False

# The splitting logic is preserved for flexibility, but by default all data goes into 'trn'.
# This is because the training script performs its own internal splitting.
# Set do_real_split = True if you want to generate separate files and pickles for each split here.
if do_real_split:
    splits = {
        'trn': lines[:n_train],
        'val': lines[n_train:n_train+n_val],
        'tst': lines[n_train+n_val:]
    }
else:
    splits = {
        'trn': lines,
        'val': [],
        'tst': []
    }

# track which files have been created
created_files = set()
print(f"Splitting complete. Splits saved to: {subfolder}")
for split, split_lines in splits.items():
    # if a split is not empty
    if len(split_lines) > 0:
        split_filename = f"{base}_filtered_{split}{ext}"
        split_filepath = os.path.join(subfolder, split_filename)
        with open(split_filepath, 'w', encoding='utf-8') as f:
            f.writelines(split_lines)
        print(f"Saved {split} set ({len(split_lines)} lines) to: {split_filepath}")
        created_files.add(split_filename)
    else:
        print(f"No lines for {split} set")

# --- Generate samples for each split file ---
import pickle
from tqdm import tqdm
import torch
import numpy as np

def char_to_binary(char):
    ascii_val = ord(char) & 127
    return np.array([int(b) for b in format(ascii_val, '08b')], dtype=np.uint8)

def generate_samples(input_path, stub_pad=CONTEXT_SIZE, split_name=None):
    features = []
    labels = []
    if split_name is not None:
        pkl_dir = subfolder
    # Count total lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    with open(input_path, 'r', encoding='utf-8') as f:
        sample_idx = 0
        for line in tqdm(f, total=total_lines, desc=f"Processing {split_name}"):
            line = line.rstrip('\n')
            # Find the position after the second '>'
            idx1 = line.find('>')
            idx2 = line.find('>', idx1 + 1) if idx1 != -1 else -1
            if idx2 == -1:
                continue  # Skip lines without two '>'
            stub = line[:idx2+1]
            seq = line[idx2+1:]
            seq = seq.strip()
            stub_padded = stub.rjust(stub_pad, ' ')
            curr = stub_padded
            for i, ch in enumerate(seq):
                feature = curr
                label = ch
                # For binary format: convert feature to 0/1 bits per char
                feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                features.append(feature_bits)
                labels.append(ord(label))
                curr = curr[1:] + label
                sample_idx += 1
            # Add end-of-formula sample
            feature = curr
            label = ';'
            feature_bits = np.concatenate([char_to_binary(c) for c in feature])
            features.append(feature_bits)
            labels.append(ord(label))
            sample_idx += 1
    # Save all features and labels in a single .npz file for this split
    features_np = np.stack(features)
    labels_np = np.array(labels, dtype=np.uint8)
    features_tensor = torch.from_numpy(features_np).float()
    labels_tensor = torch.from_numpy(labels_np).long()
    pkl_path = os.path.join(subfolder, f"{base}_filtered_{split_name}_dataset.pkl")
    with open(pkl_path, 'wb') as pf:
        pickle.dump({'features': features_tensor, 'labels': labels_tensor}, pf)
    print(f"Saved {features_tensor.shape[0]} samples as .pkl file for split '{split_name}' in {subfolder}")

for split in ['trn', 'val', 'tst']:
    split_filename = f"{base}_filtered_{split}{ext}"
    if split_filename not in created_files:
        print(f"No lines for {split} set")
        continue
    split_filepath = os.path.join(subfolder, split_filename)
    generate_samples(split_filepath, stub_pad=CONTEXT_SIZE, split_name=split)

print(f"Data preparation complete. Data saved to: {subfolder}")
