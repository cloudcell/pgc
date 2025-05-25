# Design: Alexander Bikeyev
# Date: 2025-04-20 

# Initialization scheme: 'a' = current, 'b' = identity for embeddings & processor blocks
INIT_SCHEME_STATE_TRANSFORM = 'b'  # Change to 'b' for identity initialization
INIT_SCHEME_ADDRESS_TRANSFORM = 'b'  # Change to 'b' for zero initialization

PATHS_TO_SHOW_IN_STATS = 128

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import json
import heapq

# CPU gradient offloading imports
import torch.cuda.amp as amp
import pickle
from torch.utils.data import Dataset

# Parse command line arguments
parser = argparse.ArgumentParser(description='Self-Organizing Brain Training')
parser.add_argument('--checkpoints', type=str, help='Path to checkpoints directory. If not specified, a timestamped directory will be created.')
parser.add_argument('--cpu', action='store_true', help='Force using CPU even if CUDA is available')
parser.add_argument('--tensorboard', type=str, default='runs', help='Path to TensorBoard log directory')
parser.add_argument('--stats_dir', type=str, default='brain_stats', help='Directory to save brain usage statistics')
# parser.add_argument('--address_dim', type=int, default=4, help='Dimensionality of the address space (default: 4)')
# add stopping criteria for validation accuracy (mandatory, i.e. required)
parser.add_argument('--val_acc_stop', type=float, default=0.99, required=True, help='Validation accuracy stopping criteria')
# add stopping criteria for training accuracy (mandatory, i.e. required)
parser.add_argument('--train_acc_stop', type=float, default=0.99, required=True, help='Training accuracy stopping criteria')
# add stopping criteria for loss (mandatory, i.e. required)
parser.add_argument('--train_loss_stop', type=float, default=0.001, required=True, help='Training loss stopping criteria')
# add stopping criterial for incorrect predictions (mandatory, i.e. required)
parser.add_argument('--train_incorrect_stop', type=int, default=0, required=True, help='Training incorrect predictions stopping criteria')
# add stopping criteria for epochs (mandatory, i.e. required)
parser.add_argument('--epochs_stop', type=int, default=1024, required=True, help='Epochs stopping criteria')
# two modes: jam and fit
parser.add_argument('--mode', type=str, default='fit', required=True, help='Mode: fit or jam')
# set all the hyperparameters
parser.add_argument('--input_size', type=int, default=784, required=True, help='Input size')
parser.add_argument('--num_classes', type=int, default=128, required=True, help='Number of classes')
parser.add_argument('--embedding_size', type=int, default=784, required=True, help='Embedding size')
parser.add_argument('--address_space_dim', type=int, default=3, required=True, help='Address space dimension')
parser.add_argument('--address_space_size', type=int, default=3, required=True, help='Address space size per dimension')
parser.add_argument('--num_jumps', type=int, default=8, required=True, help='Number of jumps')
# batch size
parser.add_argument('--batch_size', type=int, default=2048, required=False, help='Batch size')
# chunk size
parser.add_argument('--chunk_size', type=int, default=256, required=False, help='Chunk size')
# learning rate factor
parser.add_argument('--learning_rate_factor', type=float, default=0.9999999, required=False, help='Learning rate factor')

# dataset path
parser.add_argument('--dataset_path', type=str, default='./test.pkl', required=True, help='Dataset path')

args = parser.parse_args()

# Print all arguments and their values at launch
print("\nArguments used for running the model:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print()


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler('self_organizing_brain.log')
    ]
)
logger = logging.getLogger('SelfOrganizingBrain')


# Number of digits to show per epoch in file name
EPOCH_ID_DIGITS = 5

# CUDA devices
CUDA_DEVICES = [0, 1, 2, 3]  # List of CUDA devices to use for parallelization

# Set up device
device = torch.device('cpu')
if not args.cpu and torch.cuda.is_available():
    # Use the first device for the main operations
    device = torch.device(f'cuda:{CUDA_DEVICES[0]}')
    available_devices = [CUDA_DEVICES[i] for i in range(len(CUDA_DEVICES)) 
                        if i < torch.cuda.device_count()]
    
    if len(available_devices) > 0:
        device_names = [torch.cuda.get_device_name(i) for i in available_devices]
        logger.info(f'Using {len(available_devices)} GPUs: {", ".join(device_names)}')
        
        # Set CUDA device before initializing any models
        torch.cuda.set_device(CUDA_DEVICES[0])
    else:
        logger.warning('No CUDA devices available from specified list. Falling back to CPU.')
else:
    logger.info('Using CPU')

# --- Utility to retrieve model config and set globals ---
def set_globals_from_model(model):
    """Set global hyperparameter variables from a model instance."""
    config = SelfOrganizingBrain.get_config_from_model(model)
    globals()['input_size'] = config['input_size']
    globals()['num_classes'] = config['num_classes']
    globals()['embedding_size'] = config['embedding_size']
    globals()['address_space_dim'] = config['address_dim']
    globals()['address_space_size'] = config['brain_size']
    globals()['num_jumps'] = config['num_jumps']
    # Optionally: num_heads, etc.
    logger.info(f"Loaded model config: {config}")

# Constants (will be overwritten by set_globals_from_model if loading a model)
NUM_EPOCHS = args.epochs_stop #40
BATCH_SIZE = args.batch_size # 64 * len(CUDA_DEVICES) * 8 # * 4
CHUNK_SIZE = args.chunk_size # 32 * 4 * 4  # Adjust based on GPU memory

input_size = args.input_size  # Flatten the 28x28 images
num_classes = args.num_classes
embedding_size = args.embedding_size  #512  # Size of the embedding space
num_heads = 1
address_space_dim = args.address_space_dim  # Dimensionality of the address space (configurable)
address_space_size = args.address_space_size # 5  # 4 #8 #6  # Size of each dimension in the address space
brain_size = address_space_size  # Size of each dimension in the brain grid
num_jumps = args.num_jumps # 24 #5 # 3 # Number of steps through the brain
JUMP_OUT_IF_REVISITED = False

LEARNING_RATE_FACTOR = args.learning_rate_factor

# Address range constants for semi-free movement
# JUMP_MIN = -2  # Minimum address shift (negative for backward movement)
# JUMP_MAX = +2   # Maximum address shift (positive for forward movement)

class TextDataset(Dataset):
    def __init__(self, features, labels):
        # Debug: print type and dtype
        print(f"[DEBUG] features type: {type(features)}, dtype: {getattr(features, 'dtype', 'N/A')}")
        # Ensure features is a numpy array of float32
        features = np.array(features, dtype=np.float32)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load NLP dataset
print("Loading NLP dataset...")
dataset_path = os.path.join(args.dataset_path)
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

features = data['features']
labels = data['labels']

# Normalize features to [-1, 1] range like MNIST
if 0: features = features * 2 - 1  # TODO: fix this

dataset = TextDataset(features, labels)
# input_size = 784  # Same as MNIST for compatibility
num_classes = args.num_classes  # 

# Create data loaders
train_size = int(len(dataset) * 0.85)
val_size = int(len(dataset) * 0.10)
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# override train_dataset with the initial dataset for the mode jam
if args.mode == 'jam':
    train_dataset = dataset
else:
    pass

# Calculate optimal number of workers - 2 workers per GPU is usually sufficient
num_workers = min(4 * 2 * len(CUDA_DEVICES), os.cpu_count() or 1)
logger.info(f'Using {num_workers} DataLoader workers')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=True)


# Step 3: Define the Self-Organizing Brain Model
class SelfOrganizingBrain(nn.Module):
    def __init__(self, input_size=784, num_classes=args.num_classes, embedding_size=args.embedding_size, 
    brain_size=args.address_space_size, address_dim=args.address_space_dim, num_heads=1, num_jumps=args.num_jumps):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.brain_size = brain_size
        self.address_dim = address_dim
        self.num_heads = num_heads
        self.num_jumps = num_jumps
        
        # Store config for later retrieval
        self.model_config = {
            'input_size': input_size,
            'num_classes': num_classes,
            'embedding_size': embedding_size,
            'brain_size': brain_size,
            'address_dim': address_dim,
            'num_heads': num_heads,
            'num_jumps': num_jumps
        }

        # Initial embedding of input
        self.embedding = nn.Linear(input_size, embedding_size)
        if INIT_SCHEME_STATE_TRANSFORM == 'b' and input_size == embedding_size:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.eye(embedding_size))
                if self.embedding.bias is not None:
                    self.embedding.bias.zero_()
        
        # Initialize brain blocks
        blocks_shape = tuple([brain_size] * address_dim)
        self.brain_blocks = nn.ModuleList()
        
        # Calculate total number of blocks needed
        total_blocks = brain_size ** address_dim
        
        # Initialize each block
        for _ in range(total_blocks):
            block = nn.ModuleDict({
                'state_transform': nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU(),
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU()
                ),
                'address_transform': nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU(),
                    nn.Linear(embedding_size, address_dim * brain_size)
                )
            })
            if INIT_SCHEME_STATE_TRANSFORM == 'b':
                # Set state_transform Linear layers to identity weights, zero bias
                st_layers = [m for m in block['state_transform'] if isinstance(m, nn.Linear)]
                for l in st_layers:
                    if l.in_features == l.out_features:
                        with torch.no_grad():
                            l.weight.copy_(torch.eye(l.in_features))
                            if l.bias is not None:
                                l.bias.zero_()
                    else:
                        # fallback: leave as is if not square
                        pass
            # Address transform zeroing logic
            if INIT_SCHEME_ADDRESS_TRANSFORM == 'b':
                at_layers = [m for m in block['address_transform'] if isinstance(m, nn.Linear)]
                for l in at_layers:
                    with torch.no_grad():
                        l.weight.zero_()
                        if l.bias is not None:
                            l.bias.zero_()
            self.brain_blocks.append(block)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, num_classes)  # 128 classes for ASCII values
        )
        
        # Initialize statistics tracking
        self.reset_stats()
        
        # Fixed start address
        self.start_address = nn.Parameter(torch.zeros(1, self.address_dim, dtype=torch.long), requires_grad=False)
        
        # Additional transformation for absolute addressing
        self.absolute_transform = nn.Linear(embedding_size, self.address_dim * brain_size)

    @classmethod
    def get_config_from_model(cls, model):
        """Retrieve model config from an instance (including DataParallel-wrapped)."""
        base_model = model.module if hasattr(model, 'module') else model
        if hasattr(base_model, 'model_config'):
            return base_model.model_config
        # Fallback: try to get from attributes
        return {
            'input_size': getattr(base_model, 'input_size', None),
            'num_classes': getattr(base_model, 'num_classes', None),
            'embedding_size': getattr(base_model, 'embedding_size', None),
            'brain_size': getattr(base_model, 'brain_size', None),
            'address_dim': getattr(base_model, 'address_dim', None),
            'num_heads': getattr(base_model, 'num_heads', None),
            'num_jumps': getattr(base_model, 'num_jumps', None)
        }
        
        # Initialize statistics tracking
        self.reset_stats()
        
        # Fixed start address
        self.start_address = nn.Parameter(torch.zeros(1, self.address_dim, dtype=torch.long), requires_grad=False)
        
        # Additional transformation for absolute addressing
        self.absolute_transform = nn.Linear(embedding_size, self.address_dim * brain_size)
    
    def get_block_index(self, *coords):
        """Convert n-dimensional coordinates to flat index"""
        # Convert coords to integers if they're tensors
        coords = [int(coord.cpu().item() if torch.is_tensor(coord) else coord) for coord in coords]
        
        # Calculate flat index using stride multiplication
        # For 3D: index = x * (size^2) + y * size + z
        index = 0
        stride = 1
        for coord in reversed(coords):
            index += coord * stride
            stride *= self.brain_size
        return index
    
    def get_block_at_position(self, *coords):
        """Get block at the specified coordinates using direct indexing"""
        index = self.get_block_index(*coords)
        return self.brain_blocks[index]
    
    def get_blocks_for_batch(self, address_onehot):
        """
        Differentiable: Get weighted sum of all blocks for each batch using address_onehot.
        address_onehot: (batch_size, address_dim, brain_size)
        Returns: weighted block outputs for each batch item
        """
        batch_size = address_onehot.size(0)
        device = address_onehot.device
        address_dim = address_onehot.size(1)
        brain_size = address_onehot.size(2)
        num_blocks = brain_size ** address_dim

        # Utility: flatten multi-dim one-hot to block weights
        def flatten_onehot(address_onehot):
            # address_onehot: (batch_size, address_dim, brain_size)
            # Returns: (batch_size, num_blocks) one-hot/soft weights
            # Use einsum to get all possible combinations
            flat_weights = address_onehot[:, 0, :]
            for d in range(1, address_dim):
                flat_weights = torch.einsum('bi,bj->bij', flat_weights, address_onehot[:, d, :])
                flat_weights = flat_weights.reshape(batch_size, -1)
            return flat_weights  # (batch_size, num_blocks)

        block_weights = flatten_onehot(address_onehot)  # (batch_size, num_blocks)

        # Stack all block modules' outputs for all possible blocks
        # For each block, we need to process the batch state through that block
        # Here, we assume you have a 'state' tensor for the batch, and need to process through all blocks
        # For now, we'll return the block_weights and let the caller handle the weighted sum
        return block_weights  # (batch_size, num_blocks)

    # NOTE: In your forward pass, you must now process all blocks for the batch and combine outputs using block_weights.

    def process_through_block(self, state, block):
        """Process state through a block"""
        # Transform state
        transformed = block['state_transform'](state)
        
        # Normalize without in-place operation
        norm = torch.norm(state, p=2, dim=1, keepdim=True)
        normalized = transformed / (norm + 1e-6)  # Add epsilon to avoid division by zero
        
        return normalized

    def compute_next_address(self, state, block):
        """Compute next address using the block's address transform"""
        # Get logits from state
        logits = block['address_transform'](state)
        
        # Reshape logits to (batch_size, address_dim, brain_size)
        logits = logits.view(-1, self.address_dim, self.brain_size)
        
        # Apply Gumbel-Softmax along brain_size dimension for differentiable routing
        # hard=True returns one-hot, but gradients flow through soft sample
        address_onehot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=2)
        
        # Optionally, get indices if needed for downstream (non-differentiable)
        address_indices = address_onehot.argmax(dim=2)
        
        # Log the sampled address indices for debugging
        if self.training:
            logger.debug(f"Gumbel-Softmax address indices: {address_indices[0].tolist()}")
        
        return address_onehot  # or return address_indices if you need indices

    def forward(self, x, collect_stats=False, labels=None, predictions=None):
        batch_size = x.size(0)
        logger = logging.getLogger('SelfOrganizingBrain')
        debug_this = (x.device.index == 0) if x.device.type == 'cuda' else True
        
        # Flatten and embed input
        state = self.embedding(x.view(batch_size, -1))
        initial_state = state
        
        # Initialize address
        current_address = self.compute_next_address(initial_state, self.brain_blocks[0])

        # Initialize pathway tracking
        self.current_pathways = [[] for _ in range(batch_size)]

        # Main processing loop (differentiable routing)
        # Track which batch items have reached the origin (all-zero coordinates)
        exited = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        exit_states = torch.zeros_like(state)
        exit_addresses = torch.zeros_like(current_address)
        for i in range(self.num_jumps):
            # Only update non-exited items
            active_mask = ~exited
            if not active_mask.any():
                break

            # Get block weights for current address
            block_weights = self.get_blocks_for_batch(current_address)  # (batch_size, num_blocks)

            # Prepare all possible block modules
            all_blocks = self.brain_blocks  # length: num_blocks
            num_blocks = len(all_blocks)

            # Process state through all blocks for each batch
            all_block_outputs = []  # (batch_size, num_blocks, embedding_size)
            for block in all_blocks:
                block_out = self.process_through_block(state, block)  # (batch_size, embedding_size)
                all_block_outputs.append(block_out.unsqueeze(1))
            all_block_outputs = torch.cat(all_block_outputs, dim=1)

            # Weighted sum over blocks
            new_state = torch.bmm(block_weights.unsqueeze(1), all_block_outputs).squeeze(1)

            # Compute next address
            new_address = self.compute_next_address(new_state, self.brain_blocks[0])
            address_indices = new_address.argmax(dim=2)

            # Only update pathway, state, address for non-exited items
            for b in range(batch_size):
                if not exited[b]:
                    self.current_pathways[b].append(tuple(address_indices[b].tolist()))
                    self.record_block_usage(self.current_pathways[b][-1])

            # Check if any active item has reached the origin (all zeros)
            is_origin = (address_indices == 0).all(dim=1) & active_mask
            if is_origin.any():
                exit_states[is_origin] = new_state[is_origin]
                exit_addresses[is_origin] = new_address[is_origin]
                exited[is_origin] = True

            # Update state/address only for active items; freeze exited
            state = torch.where(active_mask.unsqueeze(1), new_state, state)
            current_address = torch.where(active_mask.unsqueeze(1).unsqueeze(2), new_address, current_address)

            # Residual connection if needed (optional)
            # residual_weight = i / max(1, self.num_jumps - 1)
            # state = state + residual_weight * initial_state
            # state = state + initial_state

        # For items that exited, use their exit state/address; others use the last state/address
        final_state = torch.where(exited.unsqueeze(1), exit_states, state)
        final_address = torch.where(exited.unsqueeze(1).unsqueeze(2), exit_addresses, current_address)

        # Final transformation: process through all blocks and combine by block weights
        block_weights = self.get_blocks_for_batch(final_address)  # (batch_size, num_blocks)
        all_block_outputs = []
        for block in all_blocks:
            block_out = self.process_through_block(final_state, block)  # (batch_size, embedding_size)
            block_out = self.process_through_block(state, block)  # (batch_size, embedding_size)
            all_block_outputs.append(block_out.unsqueeze(1))
        all_block_outputs = torch.cat(all_block_outputs, dim=1)  # (batch_size, num_blocks, embedding_size)
        final_output = torch.bmm(block_weights.unsqueeze(1), all_block_outputs).squeeze(1) + initial_state

        # Output layer
        outputs = self.output(final_output)
        
        # Record pathway with label and prediction
        if collect_stats and labels is not None:
            _, pred = torch.max(outputs.data, 1)
            for b in range(batch_size):
                self.record_pathway_with_label(self.current_pathways[b], labels[b], pred[b])

        return outputs

    def aggregate_stats_from_processes(self):
        """Aggregate statistics from all processes when using DataParallel"""
        # Only proceed if we're in a distributed setting
        if not isinstance(self, nn.DataParallel):
            return
        
        # Access the base model
        base_model = self.module
        
        # Since we can't directly access replicas in DataParallel, we'll use a different approach
        # We'll collect statistics during training/validation and synchronize here
        
        # We need to modify the forward method to collect statistics on each GPU
        # and then aggregate them here
        
        # For now, we'll log that this is happening
        logger.info("Aggregating statistics from all GPU processes")
        
        # The actual aggregation will happen through the collect_stats parameter in forward()
        # which will be set to True during validation

    def reset_stats(self):
        """Reset the brain block usage statistics"""
        self.block_usage_count = defaultdict(int)  # Count of each block's usage
        self.pathway_counter = Counter()  # Count of each pathway through the brain
        self.current_pathways = []  # List of pathways for the current batch
        self.all_pathways = []  # List of all pathways seen
        self.pathway_labels = defaultdict(list)  # Store labels associated with each pathway
        self.pathway_predictions = defaultdict(list)  # Store predictions for each pathway
    
    def record_block_usage(self, coords):
        """Record the usage of a brain block at the given coordinates"""
        # Convert coordinates to block IDs
        block_ids = []
        for coord in coords:
            # Ensure the coordinate is within valid range
            if coord < 0 or coord >= self.brain_size:
                raise ValueError(f"Invalid coordinate: {coord}. Must be between 0 and {self.brain_size - 1}")
            block_id = int(coord)
            block_ids.append(block_id)
        
        # Convert to tuple for counting
        coords_tuple = tuple(block_ids)
        self.block_usage_count[coords_tuple] += 1
        return coords_tuple
    
    def record_pathway(self, pathway):
        """Record a complete pathway through the brain"""
        # Convert pathway to a tuple for counting
        pathway_tuple = tuple(pathway)
        self.pathway_counter[pathway_tuple] += 1
        self.all_pathways.append(pathway_tuple)
    
    def record_pathway_with_label(self, pathway, label, prediction):
        """Record a complete pathway through the brain with its associated label and prediction"""
        # Convert pathway to a tuple for counting
        pathway_tuple = tuple(pathway)
        self.all_pathways.append(pathway_tuple)
        
        # Store the label and prediction associated with this pathway
        self.pathway_labels[pathway_tuple].append(label.item())
        self.pathway_predictions[pathway_tuple].append(prediction.item())
        
        # Increment the counter for this pathway
        self.pathway_counter[pathway_tuple] += 1

class GlobalStatisticsAggregator:
    """
    Class to aggregate and manage statistics from training and validation phases separately.
    This ensures clean separation of statistics between phases and across epochs.
    """
    def __init__(self):
        self.train_stats = {}  # Dictionary to store training statistics by epoch
        self.val_stats = {}    # Dictionary to store validation statistics by epoch
    
    def update_train_stats(self, model, epoch):
        """Update training statistics for the given epoch"""
        # Get the base model if using DataParallel
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Store comprehensive training statistics
        self.train_stats[epoch] = {
            'block_usage_count': base_model.block_usage_count.copy(),
            'pathway_counter': base_model.pathway_counter.copy(),
            'pathway_labels': {k: v.copy() for k, v in base_model.pathway_labels.items()},
            'pathway_predictions': {k: v.copy() for k, v in base_model.pathway_predictions.items()},
            'unique_blocks': len(base_model.block_usage_count),
            'unique_pathways': len(base_model.pathway_counter)
        }
        
        logger.info(f"Stored training statistics for epoch {epoch}")
    
    def update_val_stats(self, model, epoch):
        """Update validation statistics for the given epoch"""
        # Get the base model if using DataParallel
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Store comprehensive validation statistics
        self.val_stats[epoch] = {
            'block_usage_count': base_model.block_usage_count.copy(),
            'pathway_counter': base_model.pathway_counter.copy(),
            'pathway_labels': {k: v.copy() for k, v in base_model.pathway_labels.items()},
            'pathway_predictions': {k: v.copy() for k, v in base_model.pathway_predictions.items()},
            'unique_blocks': len(base_model.block_usage_count),
            'unique_pathways': len(base_model.pathway_counter)
        }
        
        logger.info(f"Stored validation statistics for epoch {epoch}")
    
    def get_train_stats(self, epoch):
        """Get training statistics for the given epoch"""
        return self.train_stats.get(epoch, {})
    
    def get_val_stats(self, epoch):
        """Get validation statistics for the given epoch"""
        return self.val_stats.get(epoch, {})
    
    def get_summary(self, epoch=None):
        """Get a summary of statistics for the given epoch or all epochs"""
        if epoch is not None:
            # Summary for specific epoch
            train_stats = self.get_train_stats(epoch)
            val_stats = self.get_val_stats(epoch)
            
            return {
                'epoch': epoch,
                'train': {
                    'unique_blocks': train_stats.get('unique_blocks', 0),
                    'unique_pathways': train_stats.get('unique_pathways', 0)
                },
                'val': {
                    'unique_blocks': val_stats.get('unique_blocks', 0),
                    'unique_pathways': val_stats.get('unique_pathways', 0)
                }
            }
        else:
            # Summary across all epochs
            summary = {}
            all_epochs = set(self.train_stats.keys()) | set(self.val_stats.keys())
            
            for e in sorted(all_epochs):
                summary[e] = self.get_summary(e)
            
            return summary

def get_latest_model_path(checkpoint_dir):
    """Get the path of the latest model checkpoint by extracting and comparing epoch numbers"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = list(Path(checkpoint_dir).glob('model_*.pt'))
    if not checkpoints:
        return None
    
    # Extract epoch number from filenames
    checkpoint_epochs = []
    for checkpoint in checkpoints:
        # Extract the epoch number from the filename
        # Format is model_TIMESTAMP_epoch_NUMBER.pt
        filename = os.path.basename(str(checkpoint))
        try:
            # Extract the part after "epoch_" and before ".pt"
            epoch_str = filename.split('epoch_')[1].split('.pt')[0]
            epoch_num = int(epoch_str)
            checkpoint_epochs.append((epoch_num, str(checkpoint)))
        except (IndexError, ValueError):
            logger.warning(f"Couldn't parse epoch number from checkpoint: {filename}")
            continue
    
    if not checkpoint_epochs:
        return None
    
    # Sort by epoch number (first element of tuple) and get the highest
    checkpoint_epochs.sort(key=lambda x: x[0])
    latest_epoch, latest_path = checkpoint_epochs[-1]
    
    logger.info(f"Found latest checkpoint at epoch {latest_epoch}")
    return latest_path

def save_model_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, timestamp):
    """Save model checkpoint"""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get the base model if using DataParallel
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Create checkpoint filename with timestamp and epoch
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{timestamp}_epoch_{epoch:0{EPOCH_ID_DIGITS}d}.pt')
    
    # Save model state, optimizer state, epoch, and loss
    checkpoint = {
        'model_state_dict': base_model.state_dict(),  # Save the base model state
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        # 'shift_sequence': model.shift_sequence,
        'epoch': epoch,
        'loss': loss,
        'model_config': base_model.model_config  # Save model config
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f'Saved checkpoint at epoch {epoch} to {checkpoint_path}')

def load_model_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    
    # Get the base model if using DataParallel
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict into the base model
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore model config if present
    if 'model_config' in checkpoint:
        base_model.model_config = checkpoint['model_config']
    
    # If using DataParallel, update the wrapped model
    if isinstance(model, nn.DataParallel):
        model.module = base_model
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    return checkpoint['epoch'], checkpoint['loss']

def log_stats_to_tensorboard(model, writer, epoch, phase='train'):
    """Log detailed brain statistics to TensorBoard"""
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Log basic statistics
    writer.add_scalar(f'Brain/{phase}/UniqueBlocks', len(base_model.block_usage_count), epoch)
    writer.add_scalar(f'Brain/{phase}/UniquePathways', len(base_model.pathway_counter), epoch)
    
    # Calculate pathway diversity (entropy)
    pathway_counts = np.array(list(base_model.pathway_counter.values()))
    if len(pathway_counts) > 0:
        pathway_probs = pathway_counts / pathway_counts.sum()
        entropy = -np.sum(pathway_probs * np.log2(pathway_probs + 1e-10))
        writer.add_scalar(f'Brain/{phase}/PathwayEntropy', entropy, epoch)
    
    try:
        # Create block usage heatmaps for each dimension pair
        dim_pairs = [(0,1), (1,2), (0,2)]  # Pairs of dimensions to visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Initialize 3D array to store usage counts
        usage_array = np.zeros((base_model.brain_size, base_model.brain_size, base_model.brain_size))
        
        # Debug info to understand what's happening with block usage
        logger.info(f"Block usage count entries: {len(base_model.block_usage_count)}")
        if len(base_model.block_usage_count) > 0:
            logger.info(f"Sample entries: {list(base_model.block_usage_count.items())[:5]}")
        
        # Count of skipped coordinates for debugging
        skipped_wrong_dim = 0
        skipped_out_of_bounds = 0
        skipped_other_error = 0
        
        # Fill the usage array - fix to properly handle coordinates as indices
        for coords, count in base_model.block_usage_count.items():
            try:
                # Make sure coords is treated as indices for a 3D array
                if len(coords) == 3:  # Only process 3D coordinates
                    # Check if coordinates are in bounds
                    if all(0 <= c < base_model.brain_size for c in coords):
                        usage_array[coords[0], coords[1], coords[2]] += count  # Accumulate instead of overwrite
                    else:
                        skipped_out_of_bounds += 1
                else:
                    skipped_wrong_dim += 1
            except IndexError as ie:
                skipped_out_of_bounds += 1
                logger.warning(f"Index error with coords {coords}: {str(ie)}")
            except Exception as e:
                skipped_other_error += 1
                logger.warning(f"Error processing coords {coords}: {str(e)}")
        
        # Log statistics about skipped coordinates
        total_skipped = skipped_wrong_dim + skipped_out_of_bounds + skipped_other_error
        if total_skipped > 0:
            logger.info(f"Skipped coordinates: {total_skipped} total")
            logger.info(f"  - Wrong dimension: {skipped_wrong_dim}")
            logger.info(f"  - Out of bounds: {skipped_out_of_bounds}")
            logger.info(f"  - Other errors: {skipped_other_error}")
        
        # Log some statistics about the usage array
        logger.info(f"Usage array non-zero elements: {np.count_nonzero(usage_array)}")
        logger.info(f"Usage array max value: {np.max(usage_array)}")
        
        # Create heatmaps for each dimension pair
        for (dim1, dim2), ax in zip(dim_pairs, axes):
            # Sum along the remaining dimension to get 2D projection
            remaining_dim = 3 - dim1 - dim2
            heatmap_data = np.sum(usage_array, axis=remaining_dim)
            
            # Plot heatmap
            im = ax.imshow(heatmap_data, cmap='hot', interpolation='nearest', origin='lower')
            plt.colorbar(im, ax=ax, label='Usage Count')
            ax.set_title(f'Dims {dim1}-{dim2}')
            ax.set_xlabel(f'Dimension {dim2}')
            ax.set_ylabel(f'Dimension {dim1}')
        
        plt.suptitle(f'Brain Block Usage Heatmaps ({phase.capitalize()} - Epoch {epoch+1})')
        plt.tight_layout()
        
        # Add to TensorBoard
        writer.add_figure(f'Brain/{phase}/BlockUsageHeatmap', fig, epoch)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not create heatmap for TensorBoard: {str(e)}")

def analyze_brain_stats(model, epoch, stats_dir, phase='train', global_stats_aggregator=None):
    """Analyze and save brain usage statistics"""
    # Aggregate statistics from all processes if using DataParallel
    if isinstance(model, nn.DataParallel):
        model.module.aggregate_stats_from_processes()
        base_model = model.module
    else:
        base_model = model
    
    # Calculate block usage from pathways
    block_usage_count = defaultdict(int)
    for pathway in base_model.all_pathways:
        for coords in pathway:
            block_usage_count[coords] += 1
    
    # Get the top 64 most used blocks
    top_blocks = heapq.nlargest(PATHS_TO_SHOW_IN_STATS, block_usage_count.items(), key=lambda x: x[1])
    
    # Print summary to console
    logger.info(f"Brain usage statistics for epoch {epoch}:")
    logger.info(f"  Total unique blocks used: {len(block_usage_count)}")
    logger.info(f"  Total unique pathways: {len(base_model.pathway_counter)}")
    
    # Print top blocks
    logger.info("Top most used blocks:")
    for coords, count in top_blocks:
        # Format coordinates as integers
        coords_str = f"({', '.join(map(str, coords))})"
        logger.info(f"  Block ID: {coords_str}, Count: {count}")
    
    # Get all pathways from both the counter and the labels dictionary to ensure we capture all
    all_pathways = set(base_model.pathway_counter.keys()) | set(base_model.pathway_labels.keys())
    
    # Count occurrences of each pathway
    pathway_counts = {}
    for pathway in all_pathways:
        # Count from both the counter and the labels dictionary
        counter_count = base_model.pathway_counter.get(pathway, 0)
        labels_count = len(base_model.pathway_labels.get(pathway, []))
        # Use the maximum of the two counts
        pathway_counts[pathway] = max(counter_count, labels_count)
    
    # Get the top 64 most common pathways
    top_pathways = heapq.nlargest(PATHS_TO_SHOW_IN_STATS, pathway_counts.items(), key=lambda x: x[1])
    
    # Prepare stats dictionary
    stats = {
        'epoch': epoch,
        'phase': phase,
        'top_blocks': [{'coords': list(coords), 'count': count} for coords, count in top_blocks],
        'top_pathways': [{'pathway': [list(coords) for coords in pathway], 'count': count} 
                         for pathway, count in top_pathways]
    }
    
    # Add label statistics for each pathway
    pathway_stats = []
    for pathway, _ in top_pathways:
        # Get labels associated with this pathway
        labels = base_model.pathway_labels.get(pathway, [])
        predictions = base_model.pathway_predictions.get(pathway, [])
        
        # Calculate the top 3 most frequent labels
        label_counter = Counter(labels)
        top_labels = label_counter.most_common(3)
        
        # Calculate accuracy for this pathway
        # Make sure we have matching pairs of labels and predictions
        valid_pairs = min(len(labels), len(predictions))
        if valid_pairs > 0:
            # Only count pairs where we have both label and prediction
            # Create paired lists of the same length for accurate comparison
            paired_labels = labels[:valid_pairs]
            paired_predictions = predictions[:valid_pairs]
            
            # Count correct predictions
            correct = sum(1 for i in range(valid_pairs) if paired_labels[i] == paired_predictions[i])
            accuracy = correct / valid_pairs
            sample_count = valid_pairs
            
            # Log the accuracy calculation for debugging
            logger.debug(f"Pathway accuracy calculation: {correct} correct out of {valid_pairs} samples = {accuracy:.4f}")
        else:
            correct = 0
            accuracy = 0
            sample_count = len(labels)  # Use label count even if no predictions
        
        # The actual count is the number of samples that used this pathway
        actual_count = len(labels)
        
        # Log the accuracy calculation for debugging
        if sample_count > 0:
            logger.info(f"Pathway accuracy: {correct}/{sample_count} = {accuracy:.2f}")
        
        pathway_stats.append({
            'pathway': [list(coords) for coords in pathway],
            'count': actual_count,  # Use the actual sample count
            'top_labels': [{'label': label, 'count': label_count} for label, label_count in top_labels],
            'accuracy': accuracy,
            'correct_count': correct,
            'sample_count': sample_count
        })
    
    # Update stats dictionary
    stats['pathway_stats'] = pathway_stats
    
    # Save stats to JSON file
    stats_file = os.path.join(stats_dir, f'brain_stats_{phase}_epoch_{epoch}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create a heatmap of block usage if it's a 2D projection (for visualization)
    if base_model.address_dim >= 2:
        # Create a 2D projection by summing over other dimensions
        heatmap_data = np.zeros((base_model.brain_size, base_model.brain_size))
        
        for coords, count in base_model.block_usage_count.items():
            # Use the first two dimensions for the heatmap
            # Coordinates are already integer indices
            x_idx = coords[0]
            y_idx = coords[1]
            heatmap_data[y_idx, x_idx] += count
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Usage Count')
        plt.title(f'Brain Block Usage Heatmap (Epoch {epoch})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Set the tick positions and labels to match brain size (0 to brain_size-1)
        plt.xticks(np.arange(0, base_model.brain_size, 1))
        plt.yticks(np.arange(0, base_model.brain_size, 1))
        
        # Set the axis limits to ensure we see the full grid
        plt.xlim(-0.5, base_model.brain_size - 0.5)
        plt.ylim(-0.5, base_model.brain_size - 0.5)
        
        # Add grid lines
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Save the heatmap
        heatmap_file = os.path.join(stats_dir, f'heatmap_epoch_{epoch}.png')
        plt.savefig(heatmap_file)
        plt.close()
    
    # Create a 3D visualization if the address space is 3D
    if base_model.address_dim >= 3:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create a figure for 3D scatter plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract coordinates and counts
            coords_list = []
            counts = []
            
            for coords, count in base_model.block_usage_count.items():
                if count > 0:  # Only include used blocks
                    # Use first 3 dimensions (already integer indices)
                    coords_list.append(coords[:3])
                    counts.append(count)
            
            if coords_list:
                x = [c[0] for c in coords_list]
                y = [c[1] for c in coords_list]
                z = [c[2] for c in coords_list]
                
                # Normalize counts for point size
                max_count = max(counts)
                sizes = [50 * (c / max_count) for c in counts]
                
                # Plot 3D scatter
                scatter = ax.scatter(x, y, z, c=counts, s=sizes, cmap='viridis', alpha=0.7)
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Usage Count')
                
                # Set labels and title
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_zlabel('Z Coordinate')
                ax.set_title(f'3D Brain Block Usage (Epoch {epoch})')
                
                # Set the axis limits to match brain size
                ax.set_xlim(0, base_model.brain_size - 1)
                ax.set_ylim(0, base_model.brain_size - 1)
                ax.set_zlim(0, base_model.brain_size - 1)
                
                # Set integer ticks
                ax.set_xticks(np.arange(0, base_model.brain_size, 1))
                ax.set_yticks(np.arange(0, base_model.brain_size, 1))
                ax.set_zticks(np.arange(0, base_model.brain_size, 1))
                
                # Save the 3D visualization
                viz_file = os.path.join(stats_dir, f'3d_viz_epoch_{epoch}.png')
                plt.savefig(viz_file)
                plt.close()
        except ImportError:
            logger.warning("Could not create 3D visualization. Make sure mpl_toolkits is installed.")
    
    # Print summary to console
    # logger.info(f"Brain usage statistics for epoch {epoch}:")
    # logger.info(f"Total unique blocks used: {len(base_model.block_usage_count)}")
    # logger.info(f"Total unique pathways: {len(base_model.pathway_counter)}")
    
    # # Print top blocks
    # logger.info("Top most used blocks:")
    # for coords, count in top_blocks:
    #     # Format coordinates as integers
    #     coords_str = f"({', '.join(map(str, coords))})"
    #     logger.info(f"  Block ID: {coords_str}, Count: {count}")
    
    # Print top pathways with label information and accuracy
    logger.info("Top most common pathways:")
    for pathway_stat in pathway_stats:
        pathway = pathway_stat['pathway']
        count = pathway_stat['count']
        accuracy = pathway_stat['accuracy']
        correct_count = pathway_stat['correct_count']
        sample_count = pathway_stat['sample_count']
        
        # Format each coordinate set in the pathway as integers
        pathway_str = " -> ".join([f"({', '.join(map(str, coords))})" for coords in pathway])
        
        # Log pathway information with labels and accuracy
        logger.info(f"  Pathway: {pathway_str}")
        logger.info(f"    Count: {count}, Accuracy: {accuracy:.2f} ({correct_count}/{sample_count})")
        
        # Format top labels properly
        labels_str = ", ".join([f"Label {label['label']}: {label['count']}" for label in pathway_stat['top_labels']])
        logger.info(f"    Top Labels: {labels_str}")

    # create a simple heatmap for tensorboard based on the BlockUsageHeatmap example elsewhere and log it into tensorboard
    
    

    
    return stats

def validate(model, val_loader, criterion, writer=None, epoch=None, global_stats_aggregator=None, stats_dir=None):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    logger = logging.getLogger('SelfOrganizingBrain')
    progress_bar = tqdm.tqdm(val_loader, desc="Validating")
    
    # Get base model
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Reset stats before validation
    base_model.reset_stats()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with stats collection
            outputs = model(inputs, collect_stats=True, labels=targets)
            loss = criterion(outputs, targets)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            
            # Update running loss
            val_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{val_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # Calculate final metrics
    final_loss = val_loss / len(val_loader)
    final_acc = 100. * correct / total
    incorrect = total - correct
    
    # Log validation results
    logger.info(f'Validation - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%, Correct: {correct}/{total}, Incorrect: {incorrect}/{total}')
    
    # Log to TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar('Epoch/ValLoss', final_loss, epoch)
        writer.add_scalar('Epoch/ValAccuracy', final_acc, epoch)
    
    # Analyze brain statistics after validation
    if epoch is not None and stats_dir is not None:
        # Calculate block usage from pathways
        block_usage_count = defaultdict(int)
        for pathway in base_model.all_pathways:
            for coords in pathway:
                block_usage_count[coords] += 1

        # Log block usage to TensorBoard as a heatmap (as was done in previous versions)
        
            
        
        # Log validation stats to console
        logger.info(f"Validation statistics for epoch {epoch}:")
        # logger.info(f"Total unique blocks used in validation: {len(block_usage_count)}")
        logger.info(f"Total unique pathways in validation: {len(base_model.pathway_counter)}")
        
        # # Log top blocks
        # if block_usage_count:
        #     top_blocks = heapq.nlargest(10, block_usage_count.items(), key=lambda x: x[1])
        #     logger.info("Top used blocks in validation:")
        #     for coords, count in top_blocks:
        #         coords_str = f"({', '.join(map(str, coords))})"
        #         logger.info(f"  Block {coords_str}: {count} uses")
        
        analyze_brain_stats(model, epoch, stats_dir, phase='val', global_stats_aggregator=global_stats_aggregator)
    
    # Log detailed statistics to TensorBoard
    if writer is not None and epoch is not None:
        log_stats_to_tensorboard(model, writer, epoch, phase='val')
    
    return final_loss, final_acc

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, 
            start_epoch=0, global_stats_aggregator=None, epochs_stop=None, val_acc_stop=None, 
            train_acc_stop=None, train_loss_stop=None, train_incorrect_stop=None):
    logger.info(f'Starting training for {epochs} epochs from epoch {start_epoch}')
    
    # Create TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer_dir = os.path.join(args.tensorboard, f'{os.path.basename(__file__)}_{timestamp}_{input_size}i_{embedding_size}e_{num_classes}c_{address_space_dim}d_{brain_size}s_{num_jumps}j')
    writer = SummaryWriter(writer_dir)
    logger.info(f'TensorBoard logs will be saved to {writer_dir}')
    
    # Create stats directory with timestamp
    stats_dir = os.path.join(args.stats_dir, f'stats_{timestamp}')
    os.makedirs(stats_dir, exist_ok=True)
    logger.info(f'Brain statistics will be saved to {stats_dir}')
    
    # Get base model (for statistics tracking)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 3  # Number of epochs with no improvement after which training will be stopped
    patience_counter = 0
    
    # Initialize lists to store tracking metrics
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    
    # Use tqdm for progress bar
    epoch_range = tqdm.trange(start_epoch, epochs, desc=f"Training")
    
    # Get the learning rate from the optimizer
    current_lr = optimizer.param_groups[0]['lr']
    
    
    # Main training loop
    for epoch in epoch_range:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Reset statistics for the epoch
        base_model.reset_stats()
        
        # Progress bar for batches
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Training loop for batches
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            try:
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs, collect_stats=True, labels=targets)
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                batch_total = targets.size(0)
                batch_correct = (predicted == targets).sum().item()
                total += batch_total
                correct += batch_correct
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
                
                # Log batch metrics to TensorBoard (every 10 batches)
                if batch_idx % 10 == 0:
                    # Calculate batch step for consistent x-axis in TensorBoard
                    step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Batch/Loss', running_loss/(batch_idx+1), step)
                    running_acc = 100.0 * correct / total
                    writer.add_scalar('Batch/Accuracy', running_acc, step)
                    writer.add_scalar('Batch/LearningRate', optimizer.param_groups[0]['lr'], step)
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                # Continue with the next batch instead of crashing
                continue
        
        # Log epoch statistics
        epoch_loss = running_loss / max(len(train_loader), 1)  # Avoid division by zero
        epoch_acc = 100 * correct / max(total, 1)  # Avoid division by zero
        incorrect = total - correct
        logger.info(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Correct: {correct}/{total}, Incorrect: {incorrect}/{total}')
        
        # Update global statistics aggregator with training stats
        if global_stats_aggregator is not None:
            global_stats_aggregator.update_train_stats(model, epoch)
        
        # Analyze and save training statistics
        stats = analyze_brain_stats(model, epoch + 1, stats_dir, phase='train', global_stats_aggregator=global_stats_aggregator)
        
        # Log training stats to console
        # logger.info(f"Training statistics for epoch {epoch+1}:")
        # # logger.info(f"Total unique blocks used in training: {len(base_model.block_usage_count)}")
        # logger.info(f"Total unique pathways in training: {len(base_model.pathway_counter)}")
        
        # Add brain stats to TensorBoard
        writer.add_scalar('Brain/UniqueBlocksUsed', len(base_model.block_usage_count), epoch)
        writer.add_scalar('Brain/UniquePathways', len(base_model.pathway_counter), epoch)
        
        # Record pathway diversity (entropy)
        pathway_counts = np.array(list(base_model.pathway_counter.values()))
        pathway_probs = pathway_counts / pathway_counts.sum()
        entropy = -np.sum(pathway_probs * np.log2(pathway_probs + 1e-10))
        writer.add_scalar('Brain/PathwayEntropy', entropy, epoch)
        
        # Log detailed brain statistics to TensorBoard
        log_stats_to_tensorboard(model, writer, epoch, phase='train')
        
        # Run validation to get a better measure of model performance
        val_loss, val_acc = validate(model, val_loader, criterion, writer, epoch, global_stats_aggregator, stats_dir)
        
        # Update global statistics aggregator with validation stats
        if global_stats_aggregator is not None:
            global_stats_aggregator.update_val_stats(model, epoch)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Epoch/TrainLoss', epoch_loss, epoch)
        writer.add_scalar('Epoch/TrainAccuracy', epoch_acc, epoch)
        writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
        writer.add_scalar('Epoch/ValAccuracy', val_acc, epoch)
        
        # Step the scheduler based on validation loss
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Check if learning rate changed
            if old_lr != new_lr:
                logger.info(f'Learning rate changed: {old_lr:.6f} -> {new_lr:.6f}')
            else:
                logger.info(f'Current learning rate: {new_lr:.6f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{timestamp}_epoch_{epoch+1:0{EPOCH_ID_DIGITS}d}.pt')  
        save_model_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, checkpoint_dir, timestamp)
        logger.info(f'Saved checkpoint at epoch {epoch+1} to {checkpoint_path}') #TODO: cleanup: this is a duplicate message (redundant)

        # --- EARLY STOPPING LOGIC ---
        # for mode 'fit' use the default stopping criteria
        if args.mode == 'fit':
            stop_training = False
            stop_reasons = []
            # Stop if specified number of epochs (epochs_stop) reached
            if epochs_stop is not None and (epoch + 1) >= epochs_stop:
                stop_training = True
                stop_reasons.append(f"Reached user-specified maximum epochs ({epochs_stop})")
            # Stop if validation accuracy threshold met
            if val_acc_stop is not None and val_acc >= val_acc_stop:
                stop_training = True
                stop_reasons.append(f"Validation accuracy {val_acc:.4f} >= threshold {val_acc_stop}")
            # Stop if training accuracy threshold met
            if train_acc_stop is not None and epoch_acc >= train_acc_stop:
                stop_training = True
                stop_reasons.append(f"Training accuracy {epoch_acc:.4f} >= threshold {train_acc_stop}")
            # Stop if training loss threshold met
            if train_loss_stop is not None and epoch_loss <= train_loss_stop:
                stop_training = True
                stop_reasons.append(f"Training loss {epoch_loss:.6f} <= threshold {train_loss_stop}")
            # Stop if incorrect predictions threshold met
            if train_incorrect_stop is not None and train_incorrect_stop > 0 and incorrect <= train_incorrect_stop:
                stop_training = True
                stop_reasons.append(f"Incorrect predictions {incorrect} <= threshold {train_incorrect_stop}")
            if stop_training:
                logger.info("Early stopping triggered. Reasons:")
                for reason in stop_reasons:
                    logger.info(f"  - {reason}")
                break

        # only if  the mode is 'jam' use joint conditions for stopping
        if args.mode == 'jam':
            # Get stopping criteria from args
            train_incorrect_stop = args.train_incorrect_stop
            train_loss_stop = args.train_loss_stop
            
            if incorrect == train_incorrect_stop and epoch_loss < train_loss_stop:
                logger.info(f"Stopping training: incorrect == {train_incorrect_stop} and loss < {train_loss_stop} (loss={epoch_loss:.4f})")
                break
            elif incorrect == train_incorrect_stop:
                logger.info(f"Not stopping: incorrect == {train_incorrect_stop} but loss >= {train_loss_stop} (loss={epoch_loss:.4f})")
            elif epoch_loss < train_loss_stop:
                logger.info(f"Not stopping: loss < {train_loss_stop} but incorrect > {train_incorrect_stop} (incorrect={incorrect})")

        epoch += 1
    
    writer.close()
    return model

# Create checkpoint directory with timestamp
from datetime import datetime
import re
# Dynamically generate the timestamp for the checkpoints folder at run time
checkpoint_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
if args.checkpoints:
    checkpoint_dir = args.checkpoints
    logger.info(f'Using provided checkpoint directory: {checkpoint_dir}')
else:
    checkpoint_dir = f'checkpoints_{checkpoint_timestamp}'
    logger.info(f'Using new timestamped checkpoint directory: {checkpoint_dir}')
    logger.info(f'Created new checkpoint directory: {checkpoint_dir}')

os.makedirs(checkpoint_dir, exist_ok=True)


# Step 4: Initialize the Model
model = SelfOrganizingBrain(
    input_size = input_size,
    embedding_size=embedding_size,
    brain_size=brain_size, 
    address_dim=address_space_dim,
    num_heads=num_heads,
    num_jumps=num_jumps
)

# Ensure model_config is set for checkpoint saving/loading
model.model_config = {
    'input_size': input_size,
    'num_classes': num_classes,
    'embedding_size': embedding_size,
    'brain_size': brain_size,
    'address_dim': address_space_dim,
    'num_heads': num_heads,
    'num_jumps': num_jumps
}

# Move model to primary device first
model = model.to(device)

# Multi-GPU setup with DataParallel if multiple GPUs are available
if not args.cpu and torch.cuda.is_available() and len(CUDA_DEVICES) > 1:
    available_devices = [CUDA_DEVICES[i] for i in range(len(CUDA_DEVICES)) 
                         if i < torch.cuda.device_count()]
    
    if len(available_devices) > 1:
        logger.info(f'Using DataParallel across {len(available_devices)} GPUs with CPU gradient offloading')
        model = nn.DataParallel(model, device_ids=available_devices)
    else:
        logger.info(f'Using single GPU with CPU gradient offloading')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use a learning rate scheduler that reduces LR on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=LEARNING_RATE_FACTOR,  # Reduce LR by 10% when triggered
    patience=0,  # Wait for 3 epochs with no improvement
    # verbose=True,
    threshold=0.0,  # Any non-improvement triggers reduction
    threshold_mode='abs',  # Use absolute threshold
    min_lr=1e-12  # Don't reduce below this value
)

# Load latest model if exists
latest_checkpoint = get_latest_model_path(checkpoint_dir)
start_epoch = 0
start_loss = float('inf')
if latest_checkpoint:
    logger.info(f'Found existing checkpoint: {latest_checkpoint}')
    start_epoch, start_loss = load_model_checkpoint(model, optimizer, scheduler, latest_checkpoint)

# Create global statistics aggregator
global_stats_aggregator = GlobalStatisticsAggregator()

# Train the model
trained_model = train(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    epochs=NUM_EPOCHS,
    start_epoch=start_epoch,
    global_stats_aggregator=global_stats_aggregator,
    epochs_stop=args.epochs_stop,
    val_acc_stop=args.val_acc_stop,
    train_acc_stop=args.train_acc_stop,
    train_loss_stop=args.train_loss_stop,
    train_incorrect_stop=args.train_incorrect_stop
)

# Print final statistics summary
logger.info("Training complete. Final statistics summary:")
summary = global_stats_aggregator.get_summary()
for epoch, stats in summary.items():
    logger.info(f"Epoch {epoch}:")
    logger.info(f"  Training: {stats['train']['unique_blocks']} unique blocks, {stats['train']['unique_pathways']} unique pathways")
    logger.info(f"  Validation: {stats['val']['unique_blocks']} unique blocks, {stats['val']['unique_pathways']} unique pathways")

# Step 5: Evaluate the Model
# Create evaluation stats directory with timestamp
eval_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
eval_stats_dir = os.path.join(args.stats_dir, f'stats_{eval_timestamp}')

def evaluate(model, test_loader, stats_dir=None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    logger.info("Starting evaluation")
    progress_bar = tqdm.tqdm(test_loader, desc="Evaluating")
    
    # Create stats directory if provided
    if stats_dir is not None:
        os.makedirs(stats_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    incorrect = total - correct
    final_accuracy = 100 * correct / total
    logger.info(f'Evaluation complete - Accuracy: {final_accuracy:.2f}%, Correct: {correct}/{total}, Incorrect: {incorrect}/{total}')
    
    # Create and save confusion matrix
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar(label='Usage Count')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save confusion matrix
        if stats_dir is not None:
            confusion_matrix_file = os.path.join(stats_dir, 'confusion_matrix.png')
            plt.savefig(confusion_matrix_file)
            logger.info(f'Saved confusion matrix to {confusion_matrix_file}')
        else:
            plt.savefig('confusion_matrix.png')
            logger.info('Saved confusion matrix to confusion_matrix.png')
        
        # Print classification report
        report = classification_report(all_labels, all_preds)
        logger.info(f'Classification Report:\n{report}')
    except ImportError:
        logger.warning('Could not generate confusion matrix. Make sure matplotlib, numpy, and scikit-learn are installed.')
    
    return final_accuracy

evaluate(trained_model, test_loader, stats_dir=eval_stats_dir)
