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

# Read all filtered lines as binary
with open(output_filepath, 'rb') as f:
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
            f.writelines(line.decode('latin-1') for line in split_lines)
        print(f"Saved {split} set ({len(split_lines)} lines) to: {split_filepath}")
        created_files.add(split_filename)
    else:
        print(f"No lines for {split} set")

# --- Generate samples for each split file ---
import pickle
from tqdm import tqdm
import torch
import numpy as np

def encode_chunk(chunk, chunk_type):
    if chunk_type == 'unigram':
        val = ord(chunk[0]) & 0xFF
        return np.array([int(b) for b in format(val, '08b')], dtype=np.uint8)
    elif chunk_type == 'bigram':
        c1 = ord(chunk[0]) if len(chunk) > 0 else 0
        c2 = ord(chunk[1]) if len(chunk) > 1 else 0
        val = ((c1 & 0xFF) << 8) | (c2 & 0xFF)
        return np.array([int(b) for b in format(val, '016b')], dtype=np.uint8)
    elif chunk_type == 'trigram':
        c1 = ord(chunk[0]) if len(chunk) > 0 else 0
        c2 = ord(chunk[1]) if len(chunk) > 1 else 0
        c3 = ord(chunk[2]) if len(chunk) > 2 else 0
        val = ((c1 & 0xFF) << 16) | ((c2 & 0xFF) << 8) | (c3 & 0xFF)
        return np.array([int(b) for b in format(val, '024b')], dtype=np.uint8)
    else:
        raise ValueError(f"Unknown chunk type: {chunk_type}")

def generate_samples(input_path, stub_pad=CONTEXT_SIZE, split_name=None, chunk_type='unigram'):
    features = []
    labels = []
    chunk_size = {'unigram': 1, 'bigram': 2, 'trigram': 3}[chunk_type]
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
            # Encode features and labels in chunks
            i = 0
            while i < len(seq):
                feature = curr
                label_chunk = seq[i:i+chunk_size]
                if len(label_chunk) < chunk_size:
                    # Pad label_chunk with ';' if not enough chars
                    label_chunk = label_chunk + ';' * (chunk_size - len(label_chunk))
                feature_bits = np.concatenate([encode_chunk(c, chunk_type) for c in feature]) if chunk_type == 'unigram' else np.concatenate([encode_chunk(feature[j:j+chunk_size], chunk_type) for j in range(0, len(feature), chunk_size)])
                # For label, encode as int (for bigram/trigram, pack into int)
                if chunk_type == 'unigram':
                    label_val = ord(label_chunk[0])
                elif chunk_type == 'bigram':
                    label_val = (ord(label_chunk[0]) << 8) | ord(label_chunk[1])
                elif chunk_type == 'trigram':
                    label_val = (ord(label_chunk[0]) << 16) | (ord(label_chunk[1]) << 8) | ord(label_chunk[2])
                features.append(feature_bits)
                labels.append(label_val)
                curr = curr[chunk_size:] + label_chunk
                sample_idx += 1
                i += chunk_size
            # Add end-of-formula sample
            feature = curr
            label_chunk = ';' * chunk_size
            feature_bits = np.concatenate([encode_chunk(c, chunk_type) for c in feature]) if chunk_type == 'unigram' else np.concatenate([encode_chunk(feature[j:j+chunk_size], chunk_type) for j in range(0, len(feature), chunk_size)])
            if chunk_type == 'unigram':
                label_val = ord(label_chunk[0])
            elif chunk_type == 'bigram':
                label_val = (ord(label_chunk[0]) << 8) | ord(label_chunk[1])
            elif chunk_type == 'trigram':
                label_val = (ord(label_chunk[0]) << 16) | (ord(label_chunk[1]) << 8) | ord(label_chunk[2])
            features.append(feature_bits)
            labels.append(label_val)
            sample_idx += 1
    # Save all features and labels in a single .npz file for this split
    features_np = np.stack(features)
    labels_np = np.array(labels, dtype=np.uint32)
    features_tensor = torch.from_numpy(features_np).float()
    labels_tensor = torch.from_numpy(labels_np).long()
    pkl_path = os.path.join(subfolder, f"{base}_filtered_{split_name}_dataset.pkl")
    with open(pkl_path, 'wb') as pf:
        pickle.dump({'features': features_tensor, 'labels': labels_tensor}, pf)
    print(f"Saved {features_tensor.shape[0]} samples as .pkl file for split '{split_name}' in {subfolder}")

import argparse
parser = argparse.ArgumentParser(description="Generate cheminfo dataset with chunk encoding.")
parser.add_argument('--chunk', type=str, required=True, choices=['unigram', 'bigram', 'trigram'], help='Chunk size for encoding: unigram (1 char), bigram (2 chars), trigram (3 chars)')
args = parser.parse_args()
chunk_type = args.chunk

for split in ['trn', 'val', 'tst']:
    split_filename = f"{base}_filtered_{split}{ext}"
    if split_filename not in created_files:
        print(f"No lines for {split} set")
        continue
    split_filepath = os.path.join(subfolder, split_filename)
    generate_samples(split_filepath, stub_pad=CONTEXT_SIZE, split_name=split, chunk_type=chunk_type)

print(f"Data preparation complete. Data saved to: {subfolder}")
