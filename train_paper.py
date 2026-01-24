"""
Unified training script với Paper Setup - Hỗ trợ train từ đầu hoặc resume

Usage:
    # Train từ đầu
    python train_paper.py
    
    # Resume từ checkpoint
    python train_paper.py --resume --resume_epoch 30

Paper setup:
- 80,000 training samples
- 20,000 validation samples  
- 50 epochs
- LR scheduler: halve every 5 epochs
- Weight decay: 5e-4
- Batch size: 1 (online learning)
- Optimizer: SGD
"""

import argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
from helper_fn import validate_fixed_fn, get_adj, generate_data
from networks import HGNN
import torch
from torch_geometric.data import Data
from utils.logger import Logger  # TensorBoard logging

def train_paper_setup(resume=False, resume_epoch=0, experiment_name=None):
    """Train hoặc resume training với paper configuration"""
    
    # Get script directory for relative paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Generate experiment name if not provided
    if experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'gnn_4x4_{timestamp}'
    
    # Create experiment folder path
    EXPERIMENT_FOLDER = os.path.join(SCRIPT_DIR, "experiments", experiment_name)
    
    # Create experiment folder if not exists
    if not os.path.exists(EXPERIMENT_FOLDER):
        os.makedirs(EXPERIMENT_FOLDER)
        print(f"✓ Created experiment folder: {EXPERIMENT_FOLDER}")
    
    # Paper hyperparameters
    param_dict = {
        'N': 4,
        'H': 32,
        'K': 1
    }
    
    # Generate data nếu chưa có
    print("Checking data files...")
    
    if not os.path.exists('data/train_paper_80k.npy'):
        print("Generating 80K training samples (like paper)...")
        generate_data(80000, param_dict, 'data/train_paper_80k.npy')
    
    if not os.path.exists('data/val_paper_20k.npy'):
        print("Generating 20K validation samples (like paper)...")
        generate_data(20000, param_dict, 'data/val_paper_20k.npy')
    
    # Load data
    print("Loading data...")
    train_data = np.load('data/train_paper_80k.npy')
    val_data = np.load('data/val_paper_20k.npy')
    
    print(f"Train: {train_data.shape[0]} samples")
    print(f"Val: {val_data.shape[0]} samples")
    
    # Model
    model = HGNN(param_dict['N'], param_dict['H'], param_dict['N'])
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Load checkpoint nếu resume
    start_epoch = 0
    eval_acc_list = []
    eval_loss_list = []
    
    if resume:
        checkpoint_path = os.path.join(EXPERIMENT_FOLDER, f'trained_net_paper_setup_epoch{resume_epoch}.pth')
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            print("Available checkpoints:")
            for f in sorted(os.listdir('.')):
                if f.startswith('trained_net_paper_setup_epoch') and f.endswith('.pth'):
                    print(f"  - {f}")
            return
        
        print(f"\nLoading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        print("✅ Checkpoint loaded successfully!")
        
        # Load existing log
        csv_path = os.path.join(EXPERIMENT_FOLDER, 'log_paper_setup.csv')
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            eval_acc_list = df_old['eval_accuracy'].tolist()
            eval_loss_list = df_old['eval_loss'].tolist()
            print(f"Loaded {len(eval_acc_list)} previous epoch results")
        
        start_epoch = resume_epoch
    else:
        print("\nStarting fresh training...")
        model.reset_parameters()
    
    # Optimizer với weight decay như paper
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.006,
        weight_decay=5e-4
    )
    
    # Scheduler: halve LR every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=5,
        gamma=0.5
    )
    
    # Adjust scheduler nếu resume
    if resume:
        for _ in range(start_epoch):
            scheduler.step()
    
    epochs = 50
    edge_index = get_adj(param_dict['N'])
    
    # Initialize TensorBoard logger
    log_dir = os.path.join(EXPERIMENT_FOLDER, "logs")
    logger = Logger(enable_logging=True, log_dir=log_dir)
    
    print("\n" + "="*70)
    if resume:
        print("RESUME TRAINING FROM CHECKPOINT")
        print(f"Resume from: Epoch {start_epoch}")
        print(f"Training epochs: {start_epoch+1} → {epochs}")
    else:
        print("TRAINING WITH PAPER SETUP (FROM SCRATCH)")
        print(f"Training epochs: 1 → {epochs}")
    print("="*70)
    print(f"Training samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Weight decay: 5e-4")
    print(f"Batch size: 1 (online learning)")
    print(f"Experiment: {experiment_name}")
    print(f"Experiment folder: {EXPERIMENT_FOLDER}")
    print(f"TensorBoard: {logger.writer.log_dir if logger.writer else 'disabled'}")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('Training...')
        
        model.train()
        epoch_loss = 0
        
        # Online learning - batch size = 1
        for idx in tqdm(range(len(train_data)), desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            cost_matrix = train_data[idx]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            # Hungarian ground truth
            r, c = linear_sum_assignment(cost_matrix)
            
            # Forward
            pred = model(G.x, G.edge_index)
            
            # Bidirectional loss
            loss_1 = loss_fn(pred, torch.from_numpy(c))
            d = np.argsort(c)
            loss_2 = loss_fn(pred.T, torch.from_numpy(d))
            total_loss = loss_1 + loss_2
            
            # Backward
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / len(train_data)
        
        # Validation
        print('Validating...')
        eval_acc, eval_loss = validate_fixed_fn(model, loss_fn, val_data, param_dict)
        eval_acc_list.append(eval_acc)
        eval_loss_list.append(eval_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_loss:.6f}')
        print(f'  Val Accuracy: {eval_acc*100:.2f}%')
        print(f'  Val Loss: {eval_loss:.6f}')
        
        # Log to TensorBoard
        logger.add_scalar('Train/Loss', avg_loss, epoch+1)
        logger.add_scalar('Val/Accuracy', eval_acc, epoch+1)
        logger.add_scalar('Val/Loss', eval_loss, epoch+1)
        logger.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch+1)
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, f'trained_net_paper_setup_epoch{epoch+1}.pth'))
            print(f'  → Checkpoint saved: epoch {epoch+1}')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, 'trained_net_paper_setup_final.pth'))
    
    # Close logger
    logger.close()
    
    # Save training history
    df = pd.DataFrame({
        'epoch': range(1, len(eval_acc_list)+1),
        'eval_accuracy': eval_acc_list,
        'eval_loss': eval_loss_list
    })
    df.to_csv(os.path.join(EXPERIMENT_FOLDER, 'log_paper_setup.csv'), index=False)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Final validation accuracy: {eval_acc*100:.2f}%")
    print(f"Expected to match paper: ~96-97%")
    print("Models saved to experiment folder:")
    print(f"  - {EXPERIMENT_FOLDER}/trained_net_paper_setup_final.pth")
    print(f"  - {EXPERIMENT_FOLDER}/log_paper_setup.csv")
    print(f"  - {EXPERIMENT_FOLDER}/logs/ (TensorBoard)")
    print("="*70)
    
    # Inference time measurement
    print("\n" + "="*70)
    print("MEASURING INFERENCE TIME (Paper Methodology)")
    print("="*70)
    
    import time
    from helper_fn import avoid_coll
    
    test_data = np.load('data/test_4x4.npy')
    n_runs = min(5000, len(test_data))
    
    model.eval()
    
    # Warm-up
    for _ in range(100):
        cost_matrix = test_data[0]
        x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
        G = Data(x, edge_index)
        _ = model(G.x, G.edge_index)
    
    # Timing: GNN only
    time_gnn_total = 0
    with torch.no_grad():
        for idx in range(n_runs):
            cost_matrix = test_data[idx % len(test_data)]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            t0 = time.time()
            pred = model(G.x, G.edge_index)
            time_gnn_total += (time.time() - t0)
    
    time_gnn_mean_us = (time_gnn_total / n_runs) * 1e6
    
    # Timing: GNN + Greedy
    time_gnn_greedy_total = 0
    with torch.no_grad():
        for idx in range(n_runs):
            cost_matrix = test_data[idx % len(test_data)]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            t0 = time.time()
            pred = model(G.x, G.edge_index)
            _ = avoid_coll(pred.detach().numpy(), param_dict)
            time_gnn_greedy_total += (time.time() - t0)
    
    time_gnn_greedy_mean_us = (time_gnn_greedy_total / n_runs) * 1e6
    time_greedy_only_us = time_gnn_greedy_mean_us - time_gnn_mean_us
    
    # Hungarian baseline
    time_hungarian_total = 0
    for idx in range(n_runs):
        cost_matrix = test_data[idx % len(test_data)]
        t0 = time.time()
        _ = linear_sum_assignment(cost_matrix)
        time_hungarian_total += (time.time() - t0)
    
    time_hungarian_mean_us = (time_hungarian_total / n_runs) * 1e6
    
    print("\n" + "="*70)
    print("INFERENCE TIME RESULTS (Mean over 5,000 runs)")
    print("="*70)
    print(f"\n📊 Execution Time per Sample:")
    print(f"  GNN Forward Only:        {time_gnn_mean_us:8.2f} μs")
    print(f"  Greedy Collision Only:   {time_greedy_only_us:8.2f} μs")
    print(f"  GNN + Greedy (Total):    {time_gnn_greedy_mean_us:8.2f} μs")
    print(f"  Hungarian (Baseline):    {time_hungarian_mean_us:8.2f} μs")
    print(f"\n📊 Speedup: {time_gnn_greedy_mean_us/time_hungarian_mean_us:.1f}× vs Hungarian")
    print("="*70)
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN-LSAP with Paper Setup')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=30,
                        help='Epoch number to resume from (default: 30)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: auto-generated with timestamp)')
    args = parser.parse_args()
    
    model = train_paper_setup(resume=args.resume, resume_epoch=args.resume_epoch,
                          experiment_name=args.experiment_name)
