"""
Training script với setup giống paper để đạt 96-97% accuracy

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
from helper_fn import validate_fixed_fn, get_adj, generate_data, comprehensive_test
from networks import HGNN
import torch
from torch_geometric.data import Data

def train_with_paper_setup():
    """Train model với exact paper configuration"""
    
    # Paper hyperparameters
    param_dict = {
        'N': 4,
        'H': 32,
        'K': 1
    }
    
    # Generate data như paper (nếu chưa có)
    print("Checking data files...")
    import os
    
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
    
    # Optimizer với weight decay như paper
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.006,  # Paper LR
        weight_decay=5e-4  # Paper weight decay
    )
    
    # Scheduler: halve LR every 5 epochs như paper
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=5,  # Every 5 epochs
        gamma=0.5     # Halve LR
    )
    
    epochs = 50  # Paper uses 50+
    edge_index = get_adj(param_dict['N'])
    
    print("\n" + "="*70)
    print("TRAINING WITH PAPER SETUP")
    print("="*70)
    print(f"Training samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: 0.006 (halve every 5 epochs)")
    print(f"Weight decay: 5e-4")
    print(f"Batch size: 1 (online learning)")
    print("="*70 + "\n")
    
    eval_acc_list = []
    eval_loss_list = []
    
    model.reset_parameters()
    
    for epoch in range(epochs):
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
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'trained_net_paper_setup_epoch{epoch+1}.pth')
            print(f'  → Checkpoint saved: epoch {epoch+1}')
    
    # Save final model
    torch.save(model.state_dict(), 'trained_net_paper_setup_final.pth')
    
    # Save training history
    df = pd.DataFrame({
        'epoch': range(1, epochs+1),
        'eval_accuracy': eval_acc_list,
        'eval_loss': eval_loss_list
    })
    df.to_csv('log_paper_setup.csv', index=False)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Final validation accuracy: {eval_acc*100:.2f}%")
    print(f"Expected to match paper: ~96-97%")
    print("Models saved:")
    print("  - trained_net_paper_setup_final.pth")
    print("  - log_paper_setup.csv")
    print("="*70)
    
    # =====================================================================
    # INFERENCE TIME MEASUREMENT (theo paper methodology)
    # =====================================================================
    print("\n" + "="*70)
    print("MEASURING INFERENCE TIME (Paper Methodology)")
    print("="*70)
    print("Protocol:")
    print("  - 5,000 inference runs")
    print("  - Batch size = 1 (single sample)")
    print("  - Report mean execution time (μs)")
    print("="*70 + "\n")
    
    import time
    from helper_fn import avoid_coll
    
    # Use test data for timing measurement
    test_data = np.load('data/test_4x4.npy')
    n_runs = min(5000, len(test_data))  # Use first 5000 samples
    
    model.eval()
    
    # Warm-up (để CPU cache ổn định)
    for _ in range(100):
        cost_matrix = test_data[0]
        x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
        G = Data(x, edge_index)
        _ = model(G.x, G.edge_index)
    
    # Timing: GNN forward pass only
    time_gnn_total = 0
    with torch.no_grad():
        for idx in range(n_runs):
            cost_matrix = test_data[idx % len(test_data)]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            t0 = time.time()
            pred = model(G.x, G.edge_index)
            time_gnn_total += (time.time() - t0)
    
    time_gnn_mean_us = (time_gnn_total / n_runs) * 1e6  # Convert to microseconds
    
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
    
    # Hungarian baseline for comparison
    time_hungarian_total = 0
    for idx in range(n_runs):
        cost_matrix = test_data[idx % len(test_data)]
        t0 = time.time()
        _ = linear_sum_assignment(cost_matrix)
        time_hungarian_total += (time.time() - t0)
    
    time_hungarian_mean_us = (time_hungarian_total / n_runs) * 1e6
    
    # Display results
    print("\n" + "="*70)
    print("INFERENCE TIME RESULTS (Mean over 5,000 runs)")
    print("="*70)
    print(f"\n📊 Execution Time per Sample:")
    print(f"  GNN Forward Only:        {time_gnn_mean_us:8.2f} μs")
    print(f"  Greedy Collision Only:   {time_greedy_only_us:8.2f} μs")
    print(f"  GNN + Greedy (Total):    {time_gnn_greedy_mean_us:8.2f} μs")
    print(f"  Hungarian (Baseline):    {time_hungarian_mean_us:8.2f} μs")
    
    print(f"\n📊 Speedup vs Hungarian:")
    print(f"  GNN+Greedy is {time_gnn_greedy_mean_us/time_hungarian_mean_us:.1f}× slower than Hungarian")
    
    print(f"\n📊 Component Breakdown:")
    print(f"  GNN:     {time_gnn_mean_us/time_gnn_greedy_mean_us*100:5.1f}%")
    print(f"  Greedy:  {time_greedy_only_us/time_gnn_greedy_mean_us*100:5.1f}%")
    
    # Save timing results
    timing_df = pd.DataFrame({
        'metric': ['GNN_forward_us', 'Greedy_only_us', 'GNN_Greedy_total_us', 'Hungarian_baseline_us'],
        'time_microseconds': [time_gnn_mean_us, time_greedy_only_us, time_gnn_greedy_mean_us, time_hungarian_mean_us]
    })
    timing_df.to_csv('inference_time_paper_setup.csv', index=False)
    print("\n✅ Timing results saved to: inference_time_paper_setup.csv")
    print("="*70)
    
    return model

if __name__ == '__main__':
    model = train_with_paper_setup()
