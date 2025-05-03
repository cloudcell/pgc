#!/usr/bin/env python3
"""
PGC Packer Utility

This script processes files in the following order:
1. Uuencode the input file
2. Prepend 98 '@' characters to the encoded data
3. Process the modified data with text_to_binary_dataset
4. Pack the resulting dataset with pgc_packer

Usage:
    pgc.py filename_to_pack [output_filename]

If output_filename is not provided, the input filename with '.pgc' appended will be used.
"""

import sys
import os
import binascii
import pickle
import tempfile
from text_to_binary_dataset import process_text_file, TextBinaryDataset

def uuencode_file(input_file):
    """
    Read a file and uuencode its contents.
    
    Args:
        input_file (str): Path to the input file
        
    Returns:
        str: uuencoded content
    """
    with open(input_file, 'rb') as f:
        file_data = f.read()
    
    # Process the data in chunks of 45 bytes (binascii.b2a_uu limitation)
    encoded_data = ""
    chunk_size = 45
    for i in range(0, len(file_data), chunk_size):
        chunk = file_data[i:i+chunk_size]
        encoded_chunk = binascii.b2a_uu(chunk)
        encoded_data += encoded_chunk.decode('ascii')
    
    return encoded_data

def process_to_binary_dataset(text_content):
    """
    Process text content using text_to_binary_dataset functionality.
    
    Args:
        text_content (str): Text content to process
        
    Returns:
        bytes: Pickled binary dataset
    """
    print("Processing to binary dataset...")
    
    # Create a temporary file with the text content
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(text_content)
    
    try:
        # Process the text file
        features, labels = process_text_file(temp_file_path)
        
        # Create the dataset
        dataset = TextBinaryDataset(features, labels)
        
        # Serialize the dataset to bytes
        dataset_dict = {'features': dataset.features, 'labels': dataset.labels}
        serialized_data = pickle.dumps(dataset_dict)
        
        print(f"Dataset created with {len(dataset)} samples")
        return serialized_data
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def pgc_pack(input_data, output_file):
    """
    Implement the pgc_packer functionality directly.
    
    Args:
        input_data (bytes): The binary data to pack
        output_file (str): Path to the output file
    """
    # Write the data directly to the output file
    with open(output_file, 'wb') as f:
        f.write(input_data)
    
    print(f"Successfully packed file to {output_file}")

def pack_file(input_file, output_file=None):
    """
    Process a file in the following order:
    1. Uuencode the input file
    2. Prepend 98 '@' characters to the encoded data
    3. Process the modified data with text_to_binary_dataset
    4. Pack the resulting dataset with pgc_packer
    
    Args:
        input_file (str): Path to the input file
        output_file (str, optional): Path to the output file
    """
    # Determine output filename if not provided
    if output_file is None:
        output_file = input_file + '.pgc'
    
    print(f"Packing {input_file} to {output_file}")
    
    try:
        # Check if the file exists before proceeding
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        
        # Step 1: Uuencode the input file
        print("Step 1: Uuencoding the input file...")
        encoded_content = uuencode_file(input_file)
        
        # Step 2: Prepend 98 '@' characters
        print("Step 2: Prepending 98 '@' characters...")
        modified_content = '@' * 98 + encoded_content
        
        # Step 3: Process with text_to_binary_dataset
        print("Step 3: Processing with text_to_binary_dataset...")
        binary_dataset = process_to_binary_dataset(modified_content)
        
        # Step 4: Pack with pgc_packer
        print("Step 4: Packing with pgc_packer...")
        pgc_pack(binary_dataset, output_file)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: pgc.py filename_to_pack [output_filename]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    pack_file(input_file, output_file)

if __name__ == "__main__":
    main()
