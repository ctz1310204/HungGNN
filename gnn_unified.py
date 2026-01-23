"""
Unified GNN-LSAP Script - Train & Test with Paper Setup

Usage:
    # Train from scratch
    python gnn_unified.py --mode train
    
    # Resume training
    python gnn_unified.py --mode train --resume --resume_epoch 30
    
    # Test model
    python gnn_unified.py --mode test --checkpoint trained_net_paper_setup_final.pth

Paper setup:
- 80,000 training samples
- 20,000 validation samples  
- 50 epochs
- LR scheduler: halve every 5 epochs
- Weight decay: 5e-4
- Batch size: 1 (online learning)
"""

import argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
import time
from helper_fn import validate_fixed_fn, get_adj, generate_data, avoid_coll
from networks import HGNN
import torch
from torch_geometric.data import Data
from utils.logger import Logger

def train_mode(resume=False, resume_epoch=0):
    """Training mode with paper configuration"""
    
    param_dict = {'N': 4, 'H': 32, 'K': 1}
    
    # Generate/load data
    print("Checking data files...")
    if not os.path.exists('data/train_paper_80k.npy'):
        print("Generating 80K training samples...")
        generate_data(80000, param_dict, 'data/train_paper_80k.npy')
    if not os.path.exists('data/val_paper_20k.npy'):
        print("Generating 20K validation samples...")
        generate_data(20000, param_dict, 'data/val_paper_20k.npy')
    
    train_data = np.load('data/train_paper_80k.npy')
    val_data = np.load('data/val_paper_20k.npy')
    
    # Model setup
    model = HGNN(param_dict['N'], param_dict['H'], param_dict['N'])
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Load checkpoint if resume
    start_epoch = 0
    eval_acc_list = []
    eval_loss_list = []
    train_acc_list = []  # NEW: Track training accuracy
    
    if resume:
        checkpoint_path = f'trained_net_paper_setup_epoch{resume_epoch}.pth'
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return
        model.load_state_dict(torch.load(checkpoint_path))
        start_epoch = resume_epoch
        if os.path.exists('log_paper_setup.csv'):
            df_old = pd.read_csv('log_paper_setup.csv')
            eval_acc_list = df_old['eval_accuracy'].tolist()
            eval_loss_list = df_old['eval_loss'].tolist()
            if 'train_accuracy' in df_old.columns:
                train_acc_list = df_old['train_accuracy'].tolist()
    else:
        model.reset_parameters()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.006, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    if resume:
        for _ in range(start_epoch):
            scheduler.step()
    
    epochs = 50
    edge_index = get_adj(param_dict['N'])
    logger = Logger(enable_logging=True, log_dir=None)
    
    print("\n" + "="*70)
    print("TRAINING MODE" + (" - RESUME" if resume else " - FROM SCRATCH"))
    print("="*70)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Epochs: {start_epoch+1} → {epochs}")
    print(f"TensorBoard: {logger.writer.log_dir}")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Training
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        
        for idx in tqdm(range(len(train_data)), desc=f"Training"):
            optimizer.zero_grad()
            cost_matrix = train_data[idx]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            r, c = linear_sum_assignment(cost_matrix)
            
            pred = model(G.x, G.edge_index)
            loss_1 = loss_fn(pred, torch.from_numpy(c))
            d = np.argsort(c)
            loss_2 = loss_fn(pred.T, torch.from_numpy(d))
            total_loss = loss_1 + loss_2
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
            # Calculate training accuracy
            pred_labels = torch.argmax(pred, dim=1).numpy()
            train_correct += np.sum(pred_labels == c)
            train_total += len(c)
        
        avg_loss = epoch_loss / len(train_data)
        train_acc = train_correct / train_total
        
        # Validation
        eval_acc, eval_loss = validate_fixed_fn(model, loss_fn, val_data, param_dict)
        eval_acc_list.append(eval_acc)
        eval_loss_list.append(eval_loss)
        train_acc_list.append(train_acc)
        
        print(f'  Train Loss: {avg_loss:.6f}')
        print(f'  Train Accuracy: {train_acc*100:.2f}%')  # NEW
        print(f'  Val Accuracy: {eval_acc*100:.2f}%')
        print(f'  Val Loss: {eval_loss:.6f}')
        
        # TensorBoard logging
        logger.add_scalar('Train/Loss', avg_loss, epoch+1)
        logger.add_scalar('Train/Accuracy', train_acc, epoch+1)  # NEW
        logger.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch+1)
        logger.add_scalar('Val/Accuracy', eval_acc, epoch+1)
        logger.add_scalar('Val/Loss', eval_loss, epoch+1)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'trained_net_paper_setup_epoch{epoch+1}.pth')
    
    torch.save(model.state_dict(), 'trained_net_paper_setup_final.pth')
    logger.close()
    
    # Save history
    df = pd.DataFrame({
        'epoch': range(1, len(eval_acc_list)+1),
        'train_accuracy': train_acc_list,  # NEW
        'eval_accuracy': eval_acc_list,
        'eval_loss': eval_loss_list
    })
    df.to_csv('log_paper_setup.csv', index=False)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print(f"Final train accuracy: {train_acc*100:.2f}%")
    print(f"Final val accuracy: {eval_acc*100:.2f}%")
    print("="*70)

def test_mode(checkpoint_path):
    """Testing mode"""
    
    param_dict = {'N': 4, 'H': 32, 'K': 1}
    
    # Generate test data if needed
    if not os.path.exists('data/test_4x4.npy'):
        print('Generating test data...')
        generate_data(5000, param_dict, 'data/test_4x4.npy')
    
    # Load model
    model = HGNN(param_dict['N'], param_dict['H'], param_dict['N'])
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    test_data = np.load('data/test_4x4.npy')
    edge_index = get_adj(param_dict['N'])
    
    print('='*70)
    print('TEST MODE - GNN_LSAP')
    print('='*70)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Test samples: {len(test_data)}')
    print('='*70)
    print()
    
    # Test metrics
    element_correct_before = 0
    element_correct_after = 0
    full_row_correct_before = 0
    full_row_correct_after = 0
    time_forward = 0
    time_greedy = 0
    time_hungarian = 0
    
    n_samples = len(test_data)
    N = param_dict['N']
    
    with torch.no_grad():
        for idx in range(n_samples):
            cost_matrix = test_data[idx]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            t0 = time.time()
            r, c = linear_sum_assignment(cost_matrix)
            time_hungarian += (time.time() - t0)
            
            t0 = time.time()
            pred = model(G.x, G.edge_index)
            time_forward += (time.time() - t0)
            
            pred_before = torch.argmax(pred, dim=1).numpy()
            
            t0 = time.time()
            pred_after = avoid_coll(pred.detach().numpy(), param_dict)
            time_greedy += (time.time() - t0)
            
            correct_before = np.sum(pred_before == c)
            element_correct_before += correct_before
            correct_after = np.sum(pred_after == c)
            element_correct_after += correct_after
            
            if correct_before == N:
                full_row_correct_before += 1
            if correct_after == N:
                full_row_correct_after += 1
    
    # Results
    print('ACCURACY RESULTS')
    print('='*70)
    print(f'Element Accuracy (Before Greedy): {element_correct_before/(n_samples*N)*100:.2f}%')
    print(f'Element Accuracy (After Greedy):  {element_correct_after/(n_samples*N)*100:.2f}%')
    print(f'Full Row Accuracy (Before Greedy): {full_row_correct_before/n_samples*100:.2f}%')
    print(f'Full Row Accuracy (After Greedy):  {full_row_correct_after/n_samples*100:.2f}%')
    print()
    
    print('TIMING RESULTS (Mean per sample)')
    print('='*70)
    time_forward_us = (time_forward / n_samples) * 1e6
    time_greedy_us = (time_greedy / n_samples) * 1e6
    time_hungarian_us = (time_hungarian / n_samples) * 1e6
    print(f'GNN Forward Pass:     {time_forward_us:8.2f} μs')
    print(f'Greedy Collision:     {time_greedy_us:8.2f} μs')
    print(f'GNN + Greedy (Total): {time_forward_us + time_greedy_us:8.2f} μs')
    print(f'Hungarian (Baseline): {time_hungarian_us:8.2f} μs')
    print(f'\nSpeedup vs Hungarian: {(time_forward_us + time_greedy_us) / time_hungarian_us:.2f}×')
    print('='*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN-LSAP Unified Script')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=30,
                        help='Epoch to resume from')
    parser.add_argument('--checkpoint', type=str, default='trained_net_paper_setup_final.pth',
                        help='Checkpoint path for testing')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(resume=args.resume, resume_epoch=args.resume_epoch)
    elif args.mode == 'test':
        test_mode(args.checkpoint)
