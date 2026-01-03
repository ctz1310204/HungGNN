"""
Comprehensive comparison of different LAP solving methods:
1. Hungarian Algorithm (baseline)
2. GNN Forward only
3. GNN Forward + Greedy collision avoidance

Measures both accuracy and inference time for comparison.
Uses paper methodology for timing: warm-up + 5000 runs on same sample
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from networks import HGNN
from helper_fn import get_adj, avoid_coll
from torch_geometric.data import Data
import time
from tqdm import tqdm

def test_hungarian(test_data):
    """Test Hungarian algorithm - baseline method"""
    print("\n[1/3] Testing Hungarian Algorithm...")
    N = test_data.shape[1]
    
    # Test accuracy on all samples
    print("  - Testing accuracy on 20K samples...")
    for idx in tqdm(range(len(test_data)), desc="Accuracy"):
        cost_matrix = test_data[idx]
        r, c = linear_sum_assignment(cost_matrix)
        # Hungarian always gives optimal solution
    
    # Timing: Run on 10K different samples
    print("  - Measuring inference time (10K samples)...")
    n_timing_samples = 10000
    
    # Warm-up
    for _ in range(100):
        cost_matrix = test_data[0]
        r, c = linear_sum_assignment(cost_matrix)
    
    # Actual timing (use perf_counter for higher precision)
    time_total = 0
    for idx in tqdm(range(n_timing_samples), desc="Timing"):
        cost_matrix = test_data[idx]
        t0 = time.perf_counter()
        r, c = linear_sum_assignment(cost_matrix)
        time_total += (time.perf_counter() - t0)
    
    time_us = time_total / n_timing_samples * 1e6
    return {
        'method': 'Hungarian Algorithm',
        'accuracy': 100.0,  # Always optimal
        'full_row_acc': 100.0,  # Always optimal
        'validity': 100.0,  # Always valid
        'time_us': time_us
    }

def test_gnn_forward_only(model, test_data, edge_index):
    """Test GNN forward pass only (no greedy)"""
    print("\n[2/3] Testing GNN Forward Only...")
    N = test_data.shape[1]
    
    # Test accuracy on all samples
    print("  - Testing accuracy on 20K samples...")
    element_correct = 0
    full_row_correct = 0
    valid_count = 0
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_data)), desc="Accuracy"):
            cost_matrix = test_data[idx]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            # Ground truth
            r, c = linear_sum_assignment(cost_matrix)
            
            # Forward only
            pred = model(G.x, G.edge_index)
            pred_assignments = torch.argmax(pred, dim=1).numpy()
            
            # Metrics
            element_correct += np.sum(pred_assignments == c)
            if np.array_equal(pred_assignments, c):
                full_row_correct += 1
            if len(np.unique(pred_assignments)) == N:
                valid_count += 1
    
    n_samples = len(test_data)
    elem_acc = element_correct / (n_samples * N) * 100
    full_row_acc = full_row_correct / n_samples * 100
    valid_pct = valid_count / n_samples * 100
    
    # Timing: Run on 10K different samples
    print("  - Measuring inference time (10K samples)...")
    n_timing_samples = 10000
    
    # Warm-up
    with torch.no_grad():
        for _ in range(100):
            x = torch.from_numpy(np.concatenate((test_data[0], test_data[0].T), axis=0)).float()
            G = Data(x, edge_index)
            pred = model(G.x, G.edge_index)
    
    # Actual timing (use perf_counter for higher precision)
    time_total = 0
    with torch.no_grad():
        for idx in tqdm(range(n_timing_samples), desc="Timing"):
            cost_matrix = test_data[idx]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            t0 = time.perf_counter()
            pred = model(G.x, G.edge_index)
            pred_assignments = torch.argmax(pred, dim=1).numpy()
            time_total += (time.perf_counter() - t0)
    
    time_us = time_total / n_timing_samples * 1e6
    
    return {
        'method': 'GNN Forward Only',
        'accuracy': elem_acc,
        'full_row_acc': full_row_acc,
        'validity': valid_pct,
        'time_us': time_us
    }

def test_gnn_with_greedy(model, test_data, edge_index, param_dict):
    """Test GNN forward + greedy collision avoidance"""
    print("\n[3/3] Testing GNN Forward + Greedy...")
    N = test_data.shape[1]
    
    # Test accuracy on all samples
    print("  - Testing accuracy on 20K samples...")
    element_correct = 0
    full_row_correct = 0
    valid_count = 0
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_data)), desc="Accuracy"):
            cost_matrix = test_data[idx]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            # Ground truth
            r, c = linear_sum_assignment(cost_matrix)
            
            # Forward
            pred = model(G.x, G.edge_index)
            
            # Greedy collision avoidance
            pred_assignments = avoid_coll(pred.detach().numpy(), param_dict)
            
            # Metrics
            element_correct += np.sum(pred_assignments == c)
            if np.array_equal(pred_assignments, c):
                full_row_correct += 1
            if len(np.unique(pred_assignments)) == N:
                valid_count += 1
    
    n_samples = len(test_data)
    elem_acc = element_correct / (n_samples * N) * 100
    full_row_acc = full_row_correct / n_samples * 100
    valid_pct = valid_count / n_samples * 100
    
    # Timing: Run on 10K different samples
    print("  - Measuring inference time (10K samples)...")
    n_timing_samples = 10000
    
    # Warm-up
    with torch.no_grad():
        for _ in range(100):
            x = torch.from_numpy(np.concatenate((test_data[0], test_data[0].T), axis=0)).float()
            G = Data(x, edge_index)
            pred = model(G.x, G.edge_index)
            _ = avoid_coll(pred.detach().numpy(), param_dict)
    
    # Actual timing - measure separately (use perf_counter for higher precision)
    time_forward = 0
    time_greedy = 0
    
    with torch.no_grad():
        for idx in tqdm(range(n_timing_samples), desc="Timing"):
            cost_matrix = test_data[idx]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            # Time forward pass
            t0 = time.perf_counter()
            pred = model(G.x, G.edge_index)
            time_forward += (time.perf_counter() - t0)
            
            # Time greedy avoidance
            t0 = time.perf_counter()
            pred_assignments = avoid_coll(pred.detach().numpy(), param_dict)
            time_greedy += (time.perf_counter() - t0)
    
    time_forward_us = time_forward / n_timing_samples * 1e6
    time_greedy_us = time_greedy / n_timing_samples * 1e6
    time_total_us = time_forward_us + time_greedy_us
    
    return {
        'method': 'GNN Forward + Greedy',
        'accuracy': elem_acc,
        'full_row_acc': full_row_acc,
        'validity': valid_pct,
        'time_us': time_total_us,
        'time_forward_us': time_forward_us,
        'time_greedy_us': time_greedy_us
    }

def main():
    # Load model
    param_dict = {'N': 4, 'H': 32, 'K': 1}
    model = HGNN(param_dict['N'], param_dict['H'], param_dict['N'])
    model.load_state_dict(torch.load('trained_net_paper_setup_final.pth'))
    model.eval()
    
    # Load test data
    test_data = np.load('data/test_paper_20k.npy')
    edge_index = get_adj(param_dict['N'])
    
    print(f"Model: trained_net_paper_setup_final.pth")
    print(f"Test data: {test_data.shape[0]:,} samples")
    print(f"Problem size: N={param_dict['N']}")
    
    # Run all tests
    results_hungarian = test_hungarian(test_data)
    results_gnn_only = test_gnn_forward_only(model, test_data, edge_index)
    results_gnn_greedy = test_gnn_with_greedy(model, test_data, edge_index, param_dict)
    
    # Print results
    print("\n" + "="*110)
    print("COMPARISON RESULTS - All Methods (20K test samples, 10K timing samples)")
    print("="*110)
    print(f"{'Method':<30} {'Element Acc':<15} {'Full Row Acc':<15} {'Validity':<15} {'Time (μs)'}")
    print("-"*110)
    
    # Hungarian (baseline)
    print(f"{results_hungarian['method']:<30} {results_hungarian['accuracy']:>6.2f}%        "
          f"{results_hungarian['full_row_acc']:>6.2f}%         {results_hungarian['validity']:>6.2f}%        {results_hungarian['time_us']:>8.2f}")
    
    # GNN Forward Only
    print(f"{results_gnn_only['method']:<30} {results_gnn_only['accuracy']:>6.2f}%        "
          f"{results_gnn_only['full_row_acc']:>6.2f}%         {results_gnn_only['validity']:>6.2f}%        {results_gnn_only['time_us']:>8.2f}")
    
    # GNN Forward + Greedy
    print(f"{results_gnn_greedy['method']:<30} {results_gnn_greedy['accuracy']:>6.2f}%        "
          f"{results_gnn_greedy['full_row_acc']:>6.2f}%         {results_gnn_greedy['validity']:>6.2f}%        {results_gnn_greedy['time_us']:>8.2f}")
    
    print("="*110)
    
    # Save to file
    with open('comparison_results.txt', 'w') as f:
        f.write("="*110 + "\n")
        f.write("COMPARISON RESULTS - All Methods (20K test samples, 10K timing samples)\n")
        f.write("="*110 + "\n")
        f.write(f"{'Method':<30} {'Element Acc':<15} {'Full Row Acc':<15} {'Validity':<15} {'Time (μs)'}\n")
        f.write("-"*110 + "\n")
        
        # Hungarian
        f.write(f"{results_hungarian['method']:<30} {results_hungarian['accuracy']:>6.2f}%        "
                f"{results_hungarian['full_row_acc']:>6.2f}%         {results_hungarian['validity']:>6.2f}%        {results_hungarian['time_us']:>8.2f}\n")
        
        # GNN Forward Only
        f.write(f"{results_gnn_only['method']:<30} {results_gnn_only['accuracy']:>6.2f}%        "
                f"{results_gnn_only['full_row_acc']:>6.2f}%         {results_gnn_only['validity']:>6.2f}%        {results_gnn_only['time_us']:>8.2f}\n")
        
        # GNN Forward + Greedy
        f.write(f"{results_gnn_greedy['method']:<30} {results_gnn_greedy['accuracy']:>6.2f}%        "
                f"{results_gnn_greedy['full_row_acc']:>6.2f}%         {results_gnn_greedy['validity']:>6.2f}%        {results_gnn_greedy['time_us']:>8.2f}\n")
        
        f.write("="*110 + "\n\n")
        
        # Paper comparison
        f.write("\nCOMPARISON WITH PAPER (N=4)\n")
        f.write("-"*60 + "\n")
        f.write(f"Method                  Paper Time    Our Time      Diff\n")
        f.write(f"Hungarian:              88.2 μs       {results_hungarian['time_us']:>6.2f} μs      {results_hungarian['time_us']-88.2:+6.2f} μs\n")
        f.write(f"GNN:                    353.8 μs      {results_gnn_greedy['time_us']:>6.2f} μs      {results_gnn_greedy['time_us']-353.8:+6.2f} μs\n")
        f.write("-"*60 + "\n")
        f.write(f"Paper Element Accuracy: 96.60%\n")
        f.write(f"Our Element Accuracy:   {results_gnn_greedy['accuracy']:>6.2f}%\n")
        f.write(f"Our Full Row Accuracy:  {results_gnn_greedy['full_row_acc']:>6.2f}%\n")
        f.write(f"Gap (Element):          {96.60 - results_gnn_greedy['accuracy']:>6.2f}%\n")
        f.write("-"*60 + "\n\n")
        
        # Key findings
        f.write("\nKEY FINDINGS:\n")
        f.write("-"*60 + "\n")
        f.write(f"1. Hungarian achieves 100% accuracy (optimal solution)\n")
        f.write(f"2. GNN Forward only: {results_gnn_only['accuracy']:.2f}% element acc, {results_gnn_only['full_row_acc']:.2f}% full row acc, {results_gnn_only['validity']:.2f}% validity\n")
        f.write(f"3. GNN + Greedy: {results_gnn_greedy['accuracy']:.2f}% element acc, {results_gnn_greedy['full_row_acc']:.2f}% full row acc, {results_gnn_greedy['validity']:.2f}% validity\n")
        ratio = results_gnn_greedy['time_us'] / results_hungarian['time_us']
        f.write(f"4. GNN + Greedy is {ratio:.2f}x slower than Hungarian for N=4\n")
        f.write(f"5. Trade-off: ~{100 - results_gnn_greedy['accuracy']:.2f}% accuracy loss for potential parallelization\n")
        f.write(f"6. Timing methodology: Average over 10K different samples\n")
        f.write("-"*60 + "\n")
    
    print("\n✅ Results saved to: comparison_results.txt")

if __name__ == "__main__":
    main()
