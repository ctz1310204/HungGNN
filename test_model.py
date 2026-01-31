"""
Test script for GNN_LSAP trained model
"""

import argparse
import os
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment
import time
from helper_fn import get_adj, avoid_coll, comprehensive_test
from networks import HGNN


def main():
    parser = argparse.ArgumentParser(description='Test GNN_LSAP trained model')
    parser.add_argument('--exp-name', type=str, required=True,
                       help='Experiment name (folder in experiments/)')
    parser.add_argument('--checkpoint', type=str, default='best',
                       choices=['best', 'final', 'epoch10', 'epoch20', 'epoch30', 'epoch40', 'epoch50'],
                       help='Which checkpoint to load (default: best)')
    parser.add_argument('--size', type=int, default=4,
                       help='Problem size (NxN matrix, default: 4)')
    parser.add_argument('--hidden', type=int, default=32,
                       help='Hidden channels (default: 32)')
    parser.add_argument('--test-samples', type=int, default=None,
                       help='Number of test samples (default: all available)')
    
    args = parser.parse_args()
    
    # Setup paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    exp_folder = os.path.join(SCRIPT_DIR, 'experiments', args.exp_name)
    
    if not os.path.exists(exp_folder):
        print(f"ERROR: Experiment folder not found: {exp_folder}")
        return
    
    # Model path
    if args.checkpoint == 'best':
        model_path = os.path.join(exp_folder, 'best_model.pth')
    elif args.checkpoint == 'final':
        model_path = os.path.join(exp_folder, 'trained_net_paper_setup_final.pth')
    else:
        epoch_num = args.checkpoint.replace('epoch', '')
        model_path = os.path.join(exp_folder, 'models', f'trained_net_paper_setup_epoch{epoch_num}.pth')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print(f"Available files in {exp_folder}:")
        for f in os.listdir(exp_folder):
            if f.endswith('.pth'):
                print(f"  - {f}")
        return
    
    # GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('\n' + '='*70)
    print(f'TESTING: {args.exp_name.upper()} | SIZE: {args.size}x{args.size}')
    print('='*70)
    print('COMPREHENSIVE TEST - GNN_LSAP')
    print('='*70)
    print(f'Experiment: {args.exp_name}')
    print(f'Model: {os.path.basename(model_path)}')
    print(f'Problem size: {args.size}x{args.size}')
    print(f'Device: {device}')
    print('='*70)
    print()
    
    # Load model
    param_dict = {'N': args.size, 'H': args.hidden, 'K': 1}
    model = HGNN(param_dict['N'], param_dict['H'], param_dict['N'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    # Load test data
    test_file = os.path.join(SCRIPT_DIR, 'data', f'test_{args.size}x{args.size}.npy')
    if not os.path.exists(test_file):
        print(f"WARNING: Test file not found: {test_file}")
        print(f"Generating test data...")
        from helper_fn import generate_data
        generate_data(10000 if args.test_samples is None else args.test_samples, 
                     param_dict, test_file)
    
    test_data = np.load(test_file)
    
    if args.test_samples:
        test_data = test_data[:args.test_samples]
    
    print(f'Test samples: {len(test_data)}\n')
    
    # Run comprehensive test
    loss_fn = torch.nn.CrossEntropyLoss()
    results = comprehensive_test(model, loss_fn, test_data, param_dict)
    
    # Print results
    print('\n' + '='*70)
    print('TEST RESULTS')
    print('='*70)
    print(f'Samples tested: {results["n_samples"]}')
    print()
    print('ACCURACY:')
    print(f'  Element Acc (Raw argmax):   {results["element_acc_before"]*100:6.2f}%')
    print(f'  Element Acc (After greedy): {results["element_acc_after"]*100:6.2f}%')
    print(f'  Full Row Acc (Raw):         {results["full_row_acc_before"]*100:6.2f}%')
    print(f'  Full Row Acc (After greedy):{results["full_row_acc_after"]*100:6.2f}%')
    print()
    print('PERFORMANCE:')
    print(f'  Avg Loss:            {results["avg_loss"]:.6f}')
    print(f'  Forward time:        {results["time_forward_ms"]:.3f} ms/sample')
    print(f'  Greedy time:         {results["time_greedy_ms"]:.3f} ms/sample')
    print(f'  Hungarian time:      {results["time_hungarian_ms"]:.3f} ms/sample')
    print('='*70)
    
    # Improvement stats
    element_improve = (results["element_acc_after"] - results["element_acc_before"]) * 100
    full_improve = (results["full_row_acc_after"] - results["full_row_acc_before"]) * 100
    
    print('\nGREEDY IMPROVEMENT:')
    print(f'  Element accuracy: {element_improve:+.2f}%')
    print(f'  Full row accuracy: {full_improve:+.2f}%')
    print('='*70)


if __name__ == "__main__":
    main()

