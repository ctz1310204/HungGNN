import sys
import argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import pandas as pd
from helper_fn import validate_fixed_fn, validate_random_fn, get_adj, generate_data, comprehensive_test
from networks import HGNN
import torch
from torch_geometric.data import Data

# torch.manual_seed(184)
# np.random.seed(184)

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--mode',                                    help='train or generate')
my_parser.add_argument('--id',          type=str,                   help='experiment ID')
my_parser.add_argument('--n',           type=int, default = 4,      help='LSAP problem dimension')
my_parser.add_argument('--h',           type=int, default = 32,     help='GNN hidden dimension')
my_parser.add_argument('--k',           type=int, default = 1,      help='scale factor C = k*rnd[0,1) (default 1)')
my_parser.add_argument('--s',           type=int, default = 1000,   help='number of samples to generate')
my_parser.add_argument('--fp',          type=str,                   help='path to save the generated samples')
my_parser.add_argument('--train_it',    type=int, default = 100000, help='number of training iterations')
my_parser.add_argument('--train_file',  type=str,                   help='path to the train samples')
my_parser.add_argument('--val_it',      type=int, default = 20000,  help='number of evaluation iterations')
my_parser.add_argument('--val_file',    type=str,                   help='path to the validation samples')
my_parser.add_argument('--test_file',   type=str,                   help='path to the test samples')
my_parser.add_argument('--test_chkpt',  type=str,                   help='path to the pre-trained checkpoint for test')
my_parser.add_argument('--e',           type=int, default = 50,     help='number of training epochs')
args = my_parser.parse_args()

param_dict = {
    'N': args.n,        # problem dimension
    'H': args.h,        # hidden dim
    'K': args.k      
}

loss_fn = torch.nn.CrossEntropyLoss()

model = HGNN(param_dict['N'],
             param_dict['H'],
             param_dict['N'])

optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

def start_train():
    train_iters = args.train_it
    val_iters = args.val_it
    epochs = args.e

    loss_list = []
    eval_acc_list = []
    eval_loss_list = []
    print('Start Train')
    print('LSAP dim. N = ', param_dict['N'])

    edge_index = get_adj(param_dict['N'])
    
    # Load fixed train/validation/test data
    if args.train_file is not None:
        train_data = np.load(args.train_file)
        train_iters = train_data.shape[0]
        print(f'train data {train_data.shape} loaded')
    
    if args.val_file is not None:
        val_data = np.load(args.val_file)
        val_iters = val_data.shape[0]
        print(f'validation data {val_data.shape} loaded')

    model.reset_parameters()
    for epoch in range(epochs):
       
        print(f'train epoch {epoch+1}/{epochs}...')
        
        model.train()
        
        for t_idx in tqdm(range(train_iters)):
            
            epoch_total_loss = 0
            optimizer.zero_grad()
            
            if args.train_file is not None:
                cost_matrix = train_data[t_idx]       # load train sample
            else:
                cost_matrix = np.random.random_sample((param_dict['N'], param_dict['N']))       # generate synthetic train sample
            
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()

            # build graph
            G = Data(x, edge_index)

            # compute ground truth with Hungarian alg.
            r, c = linear_sum_assignment(cost_matrix)

            # get predictions
            pred = model(G.x, G.edge_index)

            # compute loss, store and backpropagate
            loss_1 = loss_fn(pred, torch.from_numpy(c))

            d = np.argsort(c)  # column-wise truth indices (d) from row-wise ones (c)
            loss_2 = loss_fn(pred.T, torch.from_numpy(d))

            total_loss = loss_1 + loss_2

            epoch_total_loss += total_loss.item()

            total_loss.backward()

            # optionally, clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update weights
            optimizer.step()

        loss_list.append(epoch_total_loss / train_iters)

        # perform evaluation every epoch
        print('validate...')
        if args.val_file is not None:
            eval_correct, eval_loss = validate_fixed_fn(model, loss_fn, val_data, param_dict)  # validate on fixed data
        else:
            eval_correct, eval_loss = validate_random_fn(model, loss_fn, val_iters, param_dict)  # validate on random data
        eval_acc_list.append(eval_correct)
        eval_loss_list.append(eval_loss)
        print(f' Accuracy: {(100 * eval_correct):.4f}% - loss: {eval_loss:.12f}')
        print('')
        
    # save results and model
    perf_dict = {'eval_accuracy': eval_acc_list, 'eval_loss': eval_loss_list} 
    df = pd.DataFrame(perf_dict)
    df.to_csv('log_' + args.id + '.csv')
    print('log saved')
    torch.save(model.state_dict(), 'trained_net_' + args.id + '.pth')
    print('model saved')


def start_test():
    
    edge_index = get_adj(param_dict['N'])
    # Load test data
    if args.test_file is not None:
        test_data = np.load(args.test_file)
        test_iters = test_data.shape[0]
        print(f'test data {test_data.shape} loaded')
    else:
        print('test_file not found')
        sys.exit()
        
    model.load_state_dict(torch.load(args.test_chkpt))
    model.eval()
    
    print('\n' + '='*60)
    print('COMPREHENSIVE TEST - GNN_LSAP')
    print('='*60)
    
    results = comprehensive_test(model, loss_fn, test_data, param_dict)
    
    print(f'\n📊 ACCURACY METRICS:')
    print(f'  Element Accuracy (Before Greedy): {results["element_acc_before"]*100:.2f}%')
    print(f'  Element Accuracy (After Greedy):  {results["element_acc_after"]*100:.2f}%')
    print(f'  Full Row Accuracy (Before Greedy): {results["full_row_acc_before"]*100:.2f}%')
    print(f'  Full Row Accuracy (After Greedy):  {results["full_row_acc_after"]*100:.2f}%')
    
    print(f'\n⏱️  INFERENCE TIME (Average per sample):')
    print(f'  Forward Pass:      {results["time_forward_ms"]:.4f} ms')
    print(f'  Greedy Collision:  {results["time_greedy_ms"]:.4f} ms')
    print(f'  Hungarian Alg:     {results["time_hungarian_ms"]:.4f} ms')
    print(f'  Total (GNN+Greedy): {results["time_forward_ms"] + results["time_greedy_ms"]:.4f} ms')
    
    print(f'\n📈 OTHER METRICS:')
    print(f'  Average Loss: {results["avg_loss"]:.6f}')
    print(f'  Test Samples: {results["n_samples"]}')
    
    print('\n' + '='*60)
    
    # Save results to file
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'test_results_{args.id}.csv', index=False)
    print(f'Results saved to: test_results_{args.id}.csv')
    
    return results


def start_timing():
    """
    Đo timing theo paper methodology (5000 runs)
    Usage: python main.py --mode timing --test_chkpt <checkpoint> --test_file <data> --id <exp_id>
    """
    print('\n' + "="*70)
    print('INFERENCE TIME MEASUREMENT (Paper Methodology)')
    print("="*70)
    
    # Load test data
    if args.test_file is not None:
        test_data = np.load(args.test_file)
        print(f'Test data {test_data.shape} loaded')
    else:
        print("Error: Please provide --test_file")
        return
    
    # Load checkpoint
    if args.test_chkpt is not None:
        model.load_state_dict(torch.load(args.test_chkpt))
        print(f'Checkpoint {args.test_chkpt} loaded')
    else:
        print("Error: Please provide --test_chkpt")
        return
    
    print("\nProtocol:")
    print("  - 5,000 inference runs")
    print("  - Batch size = 1 (single sample)")
    print("  - Report mean execution time (μs)")
    print("="*70 + "\n")
    
    import time
    from helper_fn import avoid_coll
    
    n_runs = min(5000, len(test_data))
    edge_index = get_adj(param_dict['N'])
    model.eval()
    
    # Warm-up (100 runs để CPU cache ổn định)
    print("Warming up (100 runs)...")
    for _ in range(100):
        cost_matrix = test_data[0]
        x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
        G = Data(x, edge_index)
        _ = model(G.x, G.edge_index)
    
    # Timing: GNN forward pass only
    print(f"\nMeasuring GNN forward time ({n_runs} runs)...")
    time_gnn_total = 0
    with torch.no_grad():
        for idx in tqdm(range(n_runs), desc="GNN Forward"):
            cost_matrix = test_data[idx % len(test_data)]
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
            G = Data(x, edge_index)
            
            t0 = time.time()
            pred = model(G.x, G.edge_index)
            time_gnn_total += (time.time() - t0)
    
    time_gnn_mean_us = (time_gnn_total / n_runs) * 1e6
    
    # Timing: GNN + Greedy
    print(f"Measuring GNN + Greedy time ({n_runs} runs)...")
    time_gnn_greedy_total = 0
    with torch.no_grad():
        for idx in tqdm(range(n_runs), desc="GNN + Greedy"):
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
    print(f"Measuring Hungarian time ({n_runs} runs)...")
    time_hungarian_total = 0
    for idx in tqdm(range(n_runs), desc="Hungarian"):
        cost_matrix = test_data[idx % len(test_data)]
        t0 = time.time()
        _ = linear_sum_assignment(cost_matrix)
        time_hungarian_total += (time.time() - t0)
    
    time_hungarian_mean_us = (time_hungarian_total / n_runs) * 1e6
    
    # Display results
    print("\n" + "="*70)
    print(f"INFERENCE TIME RESULTS (Mean over {n_runs} runs)")
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
    
    # Save results
    timing_df = pd.DataFrame({
        'metric': ['GNN_forward_us', 'Greedy_only_us', 'GNN_Greedy_total_us', 'Hungarian_baseline_us'],
        'time_microseconds': [time_gnn_mean_us, time_greedy_only_us, time_gnn_greedy_mean_us, time_hungarian_mean_us]
    })
    result_file = f"inference_time_{args.id}.csv"
    timing_df.to_csv(result_file, index=False)
    print(f"\n✅ Timing results saved to: {result_file}")
    print("="*70 + "\n")
    

if __name__ == '__main__':

    if args.mode == 'train':
        start_train()
    elif args.mode == 'generate':
        generate_data(args.s, param_dict, args.fp)
    elif args.mode == 'test':
        start_test()
    elif args.mode == 'timing':
        start_timing()
    else:
        print(f'unsupported mode: {args.mode}')
