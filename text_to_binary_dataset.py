# Code for Paper: "Polymorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Design: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3


import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import sys

class TextBinaryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def char_to_binary(char):
    """Convert a character to its 8-bit binary representation."""
    # Only consider ASCII characters (0-127)
    ascii_val = ord(char) & 127
    return [int(b) for b in format(ascii_val, '08b')]

def process_text_file(file_path):
    """Process text file and convert to binary sequences with sliding windows."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Convert all valid ASCII characters to binary
    binary_data = []
    ascii_chars = []
    
    print("Converting characters to binary...")
    for char in tqdm(text, desc="Processing characters"):
        if ord(char) < 128:  # Only process ASCII characters
            binary_data.extend(char_to_binary(char))
            ascii_chars.append(char)
    
    # Create samples using sliding window
    window_size = 784  # Same as MNIST
    features = []
    labels = []
    
    # We need at least window_size binary digits plus one character for the label
    if len(binary_data) >= window_size + 8:
        # Calculate number of possible windows when sliding by 8 bits
        total_windows = (len(binary_data) - window_size) // 8
        print("\nCreating sliding windows...")
        for i in tqdm(range(total_windows), desc="Creating samples"):
            # Get window starting at i*8 (sliding by 8 bits each time)
            window_start = i * 8
            window = binary_data[window_start:window_start + window_size]
            
            # The next character after our window
            next_char_pos = (window_start + window_size) // 8
            if next_char_pos < len(ascii_chars):
                features.append(window)
                labels.append(ord(ascii_chars[next_char_pos]) & 127)  # Get ASCII value of next char
    
    return features, labels

def main():
    # Check if command-line arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python text_to_binary_dataset.py input_file output_file")
        sys.exit(1)
    
    # Get input and output files from command-line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Process the text file
    features, labels = process_text_file(input_file)
    
    # Create the dataset
    dataset = TextBinaryDataset(features, labels)
    
    # Save the dataset
    with open(output_file, 'wb') as f:
        pickle.dump({'features': dataset.features, 'labels': dataset.labels}, f)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Saved dataset to {output_file}")

if __name__ == "__main__":
    main()
