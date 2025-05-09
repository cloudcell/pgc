#!/usr/bin/env python3
"""
PGC Packer Utility

This script processes files in the following order:
1. Encode the input file (base64 or uuencode)
2. Prepend 98 '@' characters to the encoded data
3. Launch text_to_binary_dataset.py with the modified data as input
4. Launch 673_pgc_packer.py to pack things and copy the latest model from checkpoint

Usage:
    pgc.py filename_to_pack [output_filename] [--encoding uuencode|base64]

If output_filename is not provided, the input filename with '.pgc' appended will be used.
By default, base64 encoding is used. Use --encoding uuencode to use uuencode instead.
"""

import sys
import os
import tempfile
import subprocess
import shutil
import glob
import base64

def uuencode_file(input_file):
    """
    Read a file and uuencode its contents using the standard uuencode utility.
    
    Args:
        input_file (str): Path to the input file
        
    Returns:
        str: uuencoded content
    """
    # Create a temporary file to store the uuencoded output
    uu_file = input_file + '.uu'
    
    # Use the standard uuencode utility
    # The format is: uuencode input_file output_name > output_file
    subprocess.run(['uuencode', input_file, os.path.basename(input_file), '>', uu_file], 
                  shell=True, check=True)
    
    # Read the uuencoded content
    with open(uu_file, 'r') as f:
        encoded_data = f.read()
    
    # Remove the temporary file
    os.remove(uu_file)
    
    return encoded_data

def run_text_to_binary_dataset(input_file):
    """
    Launch text_to_binary_dataset.py as a separate process with the input file.
    
    Args:
        input_file (str): Path to the input file
    """
    print(f"Running text_to_binary_dataset.py on {input_file}...")
    
    # Ensure the output directory exists
    output_dir = os.path.join('data', 'NLP', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # The output path that 673_pgc_packer.py expects
    output_file = os.path.join(output_dir, 'corpus_fgmtw_dataset.pkl')
    
    # Run text_to_binary_dataset.py as a separate process
    subprocess.run([sys.executable, 'text_to_binary_dataset.py', input_file, output_file], check=True)
    
    print(f"Dataset created at {output_file}")

def run_pgc_jam(output_file):
    """
    Launch 673_pgc_jam.py to pack the dataset and copy the latest model.
    
    Args:
        output_file (str): Path to the output file
    """
    print(f"Running 673_pgc_jam.py...")
    
    # Create a checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(os.path.dirname(output_file), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Run 673_pgc_jam.py
    subprocess.run([sys.executable, '673_pgc_jam.py', '--checkpoints', checkpoint_dir], check=True)
    
    # Find the latest model in the checkpoint directory
    model_files = glob.glob(os.path.join(checkpoint_dir, 'model_*.pt'))
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        # Copy the latest model to the output file
        shutil.copy(latest_model, output_file)
        print(f"Copied latest model {latest_model} to {output_file}")
    else:
        print(f"No model files found in {checkpoint_dir}")
        sys.exit(1)

def pack_file(input_file, output_file=None, encoding='base64'):
    """
    Process a file in the following order:
    1. Encode the input file (base64 or uuencode)
    2. Prepend 98 '@' characters to the encoded data
    3. Launch text_to_binary_dataset.py with the modified data as input
    4. Launch 673_pgc_jam.py to pack things and copy the latest model
    
    Args:
        input_file (str): Path to the input file
        output_file (str, optional): Path to the output file
        encoding (str, optional): 'base64' or 'uuencode'. Default is 'base64'.
    """
    # Determine output filename if not provided
    if output_file is None:
        output_file = input_file + '.pgc'
    
    print(f"Packing {input_file} to {output_file} using {encoding} encoding")
    
    try:
        # Check if the file exists before proceeding
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        
        # Step 1: Encode the input file
        print(f"Step 1: Encoding the input file with {encoding}...")
        encoded_file = input_file + ('.b64' if encoding == 'base64' else '.uu')
        if encoding == 'base64':
            with open(input_file, 'rb') as f_in, open(encoded_file, 'wb') as f_out:
                base64.encode(f_in, f_out)
        elif encoding == 'uuencode':
            subprocess.run(f"uuencode {input_file} {os.path.basename(input_file)} > {encoded_file}", shell=True, check=True)
        else:
            print(f"Error: Unknown encoding '{encoding}'. Use 'base64' or 'uuencode'.")
            sys.exit(1)
        
        # Step 2: Prepend 98 '@' characters
        print("Step 2: Prepending 98 '@' characters...")
        with open(encoded_file, 'r') as f:
            encoded_content = f.read()
        modified_content = '@' * 98 + encoded_content
        with open(encoded_file, 'w') as f:
            f.write(modified_content)
        print(f"Created encoded file: {encoded_file}")
        
        # Step 3: Run text_to_binary_dataset.py on the modified file
        print("Step 3: Running text_to_binary_dataset.py...")
        run_text_to_binary_dataset(encoded_file)
        
        # Step 4: Run 673_pgc_packer.py and copy the latest model
        print("Step 4: Running 673_pgc_jam.py and copying latest model...")
        run_pgc_jam(output_file)
        
        print(f"Successfully packed file to {output_file}")
        print(f"Temporary file kept: {encoded_file}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: pgc.py filename_to_pack [output_filename] [--encoding uuencode|base64]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    pack_file(input_file, output_file)

if __name__ == "__main__":
    main()
