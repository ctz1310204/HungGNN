import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from networks import HGNN
import time


# adjacency matrix for bipartite graph
def get_adj(N):
    """Create edge index for bipartite graph without torch_sparse dependency."""
    b0 = torch.zeros(N, N)
    b1 = torch.ones(N, N)
    bn = torch.cat((torch.cat((b0, b1), dim=0), torch.cat((b1, b0), dim=0)), dim=1)
    
    # Convert to edge_index directly without SparseTensor
    edge_index = bn.nonzero().t().contiguous()
    return edge_index


def avoid_coll(prednp, param_dict):
    pp = np.zeros((param_dict['N'], param_dict['N']))
    minn = prednp.min()
    for elms in range(param_dict['N']):
        r1, c1 = np.where(prednp == prednp.max())
        prednp[r1, :] = np.repeat(minn, param_dict['N'])
        prednp[:, c1] = np.expand_dims(np.repeat(minn, param_dict['N']), axis=0).T
        pp[r1, c1] = 1
    return np.argmax(pp, axis=1)


def validate_fixed_fn(model, loss_fn, val_data, param_dict):
    eval_correct = 0
    eval_loss = 0
    edge_index = get_adj(param_dict['N'])
    model.eval()
    for v_idx in range(len(val_data)):
        correct = 0
        cost_matrix = val_data[v_idx]
        x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
        G = Data(x, edge_index)
        r, c = linear_sum_assignment(cost_matrix)  # truth
        pred = model(G.x, G.edge_index)  # predicted
        loss_1 = loss_fn(pred, torch.from_numpy(c))
        d = np.argsort(c)
        loss_2 = loss_fn(pred.T, torch.from_numpy(d))
        # FIX: Accumulate total loss (CrossEntropyLoss already averages over batch)
        eval_loss += (loss_1.item() + loss_2.item())
        t_idx = avoid_coll(pred.detach().numpy(), param_dict)
        # soft threshold criterion
        for a_idx in range(param_dict['N']):
            if t_idx[a_idx] == c[a_idx]:
                correct += 1
        eval_correct += (correct / param_dict['N'])
    # Average over all validation samples
    return eval_correct / len(val_data), eval_loss / len(val_data)


def validate_random_fn(model, loss_fn, val_iters, param_dict):
    eval_correct = 0
    eval_loss = 0
    edge_index = get_adj(param_dict['N'])
    model.eval()
    for v_idx in range(val_iters):
        correct = 0
        cost_matrix = np.random.random_sample((param_dict['N'], param_dict['N']))       # generate synthetic train sample
        x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
        G = Data(x, edge_index)
        r, c = linear_sum_assignment(cost_matrix)  # truth
        pred = model(G.x, G.edge_index)  # predicted
        loss_1 = loss_fn(pred, torch.from_numpy(c))
        d = np.argsort(c)
        loss_2 = loss_fn(pred.T, torch.from_numpy(d))
        eval_loss = loss_1.item() + loss_2.item()
        t_idx = avoid_coll(pred.detach().numpy(), param_dict)
        # soft threshold criterion
        for a_idx in range(param_dict['N']):
            if t_idx[a_idx] == c[a_idx]:
                correct += 1
        eval_correct += (correct / param_dict['N'])
    return eval_correct / val_iters, eval_loss / val_iters


def generate_data(n_samples, param_dict, fname):
    # TEST: Generate in [-1e10, 1e10] then normalize to [-1, 1] (like DL-based_LAP original)
    cm = np.random.uniform(-1e10, 1e10, (1, param_dict['N'], param_dict['N']))
    cm = cm / 1e10  # Normalize to [-1, 1]
    for t in tqdm(range(1, n_samples)):
        cm2 = np.random.uniform(-1e10, 1e10, (1, param_dict['N'], param_dict['N']))
        cm2 = cm2 / 1e10  # Normalize to [-1, 1]
        cm = np.concatenate((cm, cm2))
    np.save(fname, cm)
    print('done')


def comprehensive_test(model, loss_fn, test_data, param_dict):
    """Comprehensive test with multiple metrics and timing
    
    Returns:
        dict with keys:
        - element_acc_before: Element accuracy before greedy (from argmax)
        - element_acc_after: Element accuracy after greedy  
        - full_row_acc_before: Full row accuracy before greedy
        - full_row_acc_after: Full row accuracy after greedy
        - avg_loss: Average loss
        - time_forward_ms: Average forward pass time (ms)
        - time_greedy_ms: Average greedy time (ms)
        - time_hungarian_ms: Average Hungarian time (ms)
    """
    edge_index = get_adj(param_dict['N'])
    model.eval()
    
    # Metrics counters
    element_correct_before = 0
    element_correct_after = 0
    full_row_correct_before = 0
    full_row_correct_after = 0
    total_loss = 0
    
    # Timing
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
            
            # Hungarian ground truth (with timing)
            t0 = time.time()
            r, c = linear_sum_assignment(cost_matrix)
            time_hungarian += (time.time() - t0)
            
            # Forward pass (with timing)
            t0 = time.time()
            pred = model(G.x, G.edge_index)
            time_forward += (time.time() - t0)
            
            # Loss computation
            loss_1 = loss_fn(pred, torch.from_numpy(c))
            d = np.argsort(c)
            loss_2 = loss_fn(pred.T, torch.from_numpy(d))
            total_loss += (loss_1.item() + loss_2.item())
            
            # Predictions before greedy (argmax)
            pred_before = torch.argmax(pred, dim=1).numpy()
            
            # Predictions after greedy (with timing)
            t0 = time.time()
            pred_after = avoid_coll(pred.detach().numpy(), param_dict)
            time_greedy += (time.time() - t0)
            
            # Element accuracy before greedy
            correct_before = np.sum(pred_before == c)
            element_correct_before += correct_before
            
            # Element accuracy after greedy  
            correct_after = np.sum(pred_after == c)
            element_correct_after += correct_after
            
            # Full row accuracy before (all N assignments must be correct)
            if correct_before == N:
                full_row_correct_before += 1
                
            # Full row accuracy after
            if correct_after == N:
                full_row_correct_after += 1
    
    # Convert to percentages and averages
    results = {
        'element_acc_before': element_correct_before / (n_samples * N),
        'element_acc_after': element_correct_after / (n_samples * N),
        'full_row_acc_before': full_row_correct_before / n_samples,
        'full_row_acc_after': full_row_correct_after / n_samples,
        'avg_loss': total_loss / n_samples,
        'time_forward_ms': (time_forward / n_samples) * 1000,  # ms per sample
        'time_greedy_ms': (time_greedy / n_samples) * 1000,
        'time_hungarian_ms': (time_hungarian / n_samples) * 1000,
        'n_samples': n_samples
    }
    
    return results


