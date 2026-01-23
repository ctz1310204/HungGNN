"""
Test script for GNN_LSAP trained model
"""

from helper_fn import get_adj, avoid_coll
from networks import HGNN
import torch
from torch_geometric.data import Data
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# Load model
param_dict = {'N': 4, 'H': 32, 'K': 1}
model = HGNN(param_dict['N'], param_dict['H'], param_dict['N'])
model.load_state_dict(torch.load('trained_net_paper_setup_final.pth'))
model.eval()

# Load test data
test_data = np.load('data/test_4x4.npy')
edge_index = get_adj(param_dict['N'])

print('='*70)
print('COMPREHENSIVE TEST - GNN_LSAP (Paper Setup)')
print('='*70)
print(f'Model: trained_net_paper_setup_final.pth')
print(f'Training epochs: 50')
print(f'Final validation accuracy: 95.70%')
print(f'Test samples: {len(test_data)}')
print('='*70)
print()

# Metrics
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
        
        # Hungarian ground truth
        t0 = time.time()
        r, c = linear_sum_assignment(cost_matrix)
        time_hungarian += (time.time() - t0)
        
        # Forward pass
        t0 = time.time()
        pred = model(G.x, G.edge_index)
        time_forward += (time.time() - t0)
        
        # Predictions before greedy
        pred_before = torch.argmax(pred, dim=1).numpy()
        
        # Predictions after greedy
        t0 = time.time()
        pred_after = avoid_coll(pred.detach().numpy(), param_dict)
        time_greedy += (time.time() - t0)
        
        # Element accuracy before
        correct_before = np.sum(pred_before == c)
        element_correct_before += correct_before
        
        # Element accuracy after
        correct_after = np.sum(pred_after == c)
        element_correct_after += correct_after
        
        # Full row accuracy
        if correct_before == N:
            full_row_correct_before += 1
        if correct_after == N:
            full_row_correct_after += 1

# Results
element_acc_before = element_correct_before / (n_samples * N)
element_acc_after = element_correct_after / (n_samples * N)
full_row_acc_before = full_row_correct_before / n_samples
full_row_acc_after = full_row_correct_after / n_samples
time_forward_us = (time_forward / n_samples) * 1e6
time_greedy_us = (time_greedy / n_samples) * 1e6
time_hungarian_us = (time_hungarian / n_samples) * 1e6

print('ACCURACY RESULTS')
print('='*70)
print(f'Element Accuracy (Before Greedy): {element_acc_before*100:.2f}%')
print(f'Element Accuracy (After Greedy):  {element_acc_after*100:.2f}%')
print(f'Full Row Accuracy (Before Greedy): {full_row_acc_before*100:.2f}%')
print(f'Full Row Accuracy (After Greedy):  {full_row_acc_after*100:.2f}%')
print()

print('TIMING RESULTS (Mean per sample)')
print('='*70)
print(f'GNN Forward Pass:     {time_forward_us:8.2f} μs')
print(f'Greedy Collision:     {time_greedy_us:8.2f} μs')
print(f'GNN + Greedy (Total): {time_forward_us + time_greedy_us:8.2f} μs')
print(f'Hungarian (Baseline): {time_hungarian_us:8.2f} μs')
print()
print(f'Speedup vs Hungarian: {(time_forward_us + time_greedy_us) / time_hungarian_us:.2f}×')
print('='*70)
