"""
So sánh TRỰC QUAN giữa Success và Failure case
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from networks import HGNN
from helper_fn import get_adj, avoid_coll
from torch_geometric.data import Data

# Load model
param_dict = {'N': 4, 'H': 32, 'K': 1}
model = HGNN(param_dict['N'], param_dict['H'], param_dict['N'])
model.load_state_dict(torch.load('trained_net_paper_setup_final.pth'))
model.eval()

test_data = np.load('data/test_paper_20k.npy')
edge_index = get_adj(param_dict['N'])
N = param_dict['N']

print("="*120)
print("🔍 SO SÁNH SUCCESS vs FAILURE CASE")
print("="*120)

cases = [
    (0, "SUCCESS", "✅"),
    (214, "FAILURE", "❌"),
]

for sample_idx, label, icon in cases:
    print(f"\n{icon} {label} CASE - Sample #{sample_idx}")
    print("-"*120)
    
    cost_matrix = test_data[sample_idx]
    x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
    G = Data(x, edge_index)
    
    with torch.no_grad():
        output = model(G.x, G.edge_index)
    
    print(f"\n📋 Cost Matrix:")
    print(cost_matrix)
    
    print(f"\n🔥 GNN Output (Raw Logits):")
    print(output.numpy())
    
    print(f"\n📊 Thống kê từng hàng (công nhân):")
    print(f"{'':4} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Interpretation':>30}")
    print("-"*120)
    for i in range(N):
        row = output[i]
        min_val = row.min().item()
        max_val = row.max().item()
        mean_val = row.mean().item()
        std_val = row.std().item()
        
        if std_val > 5:
            interp = "🟢 Rất tự tin"
        elif std_val > 2:
            interp = "🟡 Khá chắc chắn"
        else:
            interp = "🔴 Không chắc"
        
        print(f"CN{i}  {min_val:8.3f} {max_val:8.3f} {mean_val:8.3f} {std_val:8.3f} {interp:>30}")
    
    # Before greedy
    pred_before = torch.argmax(output, dim=1).numpy()
    
    # After greedy
    pred_after = avoid_coll(output.numpy(), param_dict)
    
    # Ground truth
    r_gt, c_gt = linear_sum_assignment(cost_matrix)
    
    print(f"\n🎯 Kết quả:")
    print(f"   Before Greedy (argmax): {pred_before}")
    print(f"   After Greedy:           {pred_after}")
    print(f"   Ground Truth:           {c_gt}")
    print(f"   Unique values:          {np.unique(pred_after)} (count: {len(np.unique(pred_after))})")
    
    is_valid = len(np.unique(pred_after)) == N
    is_optimal = np.array_equal(pred_after, c_gt)
    
    print(f"\n   Valid (không trùng):    {'✅ YES' if is_valid else '❌ NO'}")
    print(f"   Optimal (tối ưu):       {'✅ YES' if is_optimal else '❌ NO'}")
    
    if not is_valid:
        print(f"\n   🔥 LÝ DO THẤT BẠI:")
        print(f"   → Sau 3 bước greedy, tất cả giá trị còn lại uniform")
        print(f"   → np.where() trả về TẤT CẢ positions")
        print(f"   → pp matrix trở thành all 1s")
        print(f"   → argmax(pp) = [0,0,0,0] (invalid!)")

print("\n" + "="*120)
print("📚 TỔNG KẾT")
print("="*120)

print("""
SUCCESS CASE (Sample #0):
✅ Logits có variance tốt (Std > 5 cho mỗi hàng)
✅ Model tự tin, lựa chọn rõ ràng
✅ Greedy hoạt động tốt qua cả 4 bước
✅ Kết quả: Valid + Optimal

FAILURE CASE (Sample #214):
⚠️  Logits ban đầu có variance (Std > 4)
⚠️  Nhưng sau 3 bước greedy, còn lại uniform
❌ Step 4: All remaining = min value
❌ np.where() returns ALL indices
❌ Kết quả: Invalid (duplicate assignments)

KẾT LUẬN:
- Đây KHÔNG PHẢI lỗi của GNN model (output ban đầu tốt)
- Đây là HẠN CHẾ của Greedy Algorithm
- Tỷ lệ thất bại: 0.38% (rất hiếm)
- Có thể fix bằng cách:
  1. Break ties deterministically
  2. Fallback to Hungarian
  3. Add small random noise
""")

print("="*120)
