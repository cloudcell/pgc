#!/usr/bin/env python3
"""
PGC Packer Utility

This script processes files in the following order:
1. Uuencode the input file
2. Prepend 98 '@' characters to the encoded data
3. Launch text_to_binary_dataset.py with the modified data as input
4. Launch 673_pgc_packer.py to pack things and copy the latest model from checkpoint

Usage:
    pgc.py filename_to_pack [output_filename]

If output_filename is not provided, the input filename with '.pgc' appended will be used.
"""

import sys
import os
import subprocess
import shutil
from datetime import datetime

# Set CUDA_VISIBLE_DEVICES to match CUDA_DEVICES in 673_pgc_packer.py
CUDA_DEVICES = [0, 1, 2, 3]  # <-- Keep in sync with 673_pgc_packer.py
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in CUDA_DEVICES)

import glob

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

def run_pgc_packer(output_file, float16=False, no_cpu_offload=False):
    """
    Launch 673_pgc_packer.py to pack the dataset and copy the latest model.
    
    Args:
        output_file (str): Path to the output file
        float16 (bool): Whether to enable float16/mixed precision training
        no_cpu_offload (bool): Whether to disable gradient offloading to CPU RAM
    """
    print(f"Running 673_pgc_packer.py...")
    
    # Create a checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(os.path.dirname(output_file), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build command
    cmd = [sys.executable, '673_pgc_packer.py', '--checkpoints', checkpoint_dir]
    if float16:
        cmd.append('--float16')
    if no_cpu_offload:
        cmd.append('--no-cpu-offload')
    subprocess.run(cmd, check=True)
    
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

def pack_file(input_file, output_file=None, float16=False, no_cpu_offload=False):
    """
    Process a file in the following order:
    1. Uuencode the input file
    2. Prepend 98 '@' characters to the encoded data
    3. Launch text_to_binary_dataset.py with the modified data as input
    4. Launch 673_pgc_packer.py to pack things and copy the latest model
    
    Args:
        input_file (str): Path to the input file
        output_file (str, optional): Path to the output file
        float16 (bool): Whether to enable float16/mixed precision training
        no_cpu_offload (bool): Whether to disable gradient offloading to CPU RAM
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
        
        # Step 1: Uuencode the input file directly to a .uu file
        print("Step 1: Uuencoding the input file...")
        uu_file = input_file + '.uu'
        
        # Run uuencode command: uuencode input_file output_name > output_file
        subprocess.run(f"uuencode {input_file} {os.path.basename(input_file)} > {uu_file}", 
                      shell=True, check=True)
        
        # Step 2: Prepend 98 '@' characters
        print("Step 2: Prepending 98 '@' characters...")
        with open(uu_file, 'r') as f:
            encoded_content = f.read()
        
        modified_content = '@' * 98 + encoded_content
        
        # Write the modified content back to the file
        with open(uu_file, 'w') as f:
            f.write(modified_content)
        
        print(f"Created uuencoded file: {uu_file}")
        
        # Step 3: Run text_to_binary_dataset.py on the modified file
        print("Step 3: Running text_to_binary_dataset.py...")
        run_text_to_binary_dataset(uu_file)
        
        # Step 4: Run 673_pgc_packer.py and copy the latest model
        print("Step 4: Running 673_pgc_packer.py and copying latest model...")
        run_pgc_packer(output_file, float16=float16, no_cpu_offload=no_cpu_offload)
        
        print(f"Successfully packed file to {output_file}")
        print(f"Temporary file kept: {uu_file}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PGC Packer Utility')
    parser.add_argument('filename_to_pack', type=str, help='Input file to pack')
    parser.add_argument('output_filename', type=str, nargs='?', default=None, help='Optional output filename')
    parser.add_argument('--float16', action='store_true', help='Enable float16/mixed precision training for 673_pgc_packer.py')
    parser.add_argument('--no-cpu-offload', action='store_true', help='Disable offloading gradients to CPU RAM (FSDP cpu_offload) for 673_pgc_packer.py')
    args = parser.parse_args()

    pack_file(args.filename_to_pack, args.output_filename, float16=args.float16, no_cpu_offload=args.no_cpu_offload)

if __name__ == "__main__":
    main()
