"""
GIẢI THÍCH CHI TIẾT GNN - TIẾNG VIỆT
Hướng dẫn từng bước cách GNN biến đổi cost matrix thành assignment
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from networks import HGNN
from helper_fn import get_adj, avoid_coll
from torch_geometric.data import Data

print("="*100)
print("🎓 HƯỚNG DẪN: GNN GIẢI BÀI TOÁN LINEAR ASSIGNMENT")
print("="*100)

# Load model
param_dict = {'N': 4, 'H': 32, 'K': 1}
model = HGNN(param_dict['N'], param_dict['H'], param_dict['N'])
model.load_state_dict(torch.load('trained_net_paper_setup_final.pth'))
model.eval()

# Load test data
test_data = np.load('data/test_paper_20k.npy')
edge_index = get_adj(param_dict['N'])
N = param_dict['N']

# Lấy 1 ví dụ thành công
cost_matrix = test_data[0]

print("\n" + "="*100)
print("📋 BÀI TOÁN: PHÂN CÔNG 4 CÔNG NHÂN CHO 4 CÔNG VIỆC")
print("="*100)

print("\n🎯 Cost Matrix (Ma trận chi phí):")
print("   Mỗi hàng = 1 công nhân")
print("   Mỗi cột = 1 công việc")
print("   Giá trị = chi phí khi công nhân i làm công việc j")
print()
for i in range(4):
    print(f"   Công nhân {i}: ", end="")
    for j in range(4):
        print(f"[CV{j}: {cost_matrix[i,j]:.3f}]  ", end="")
    print()

# Tính optimal solution
r_opt, c_opt = linear_sum_assignment(cost_matrix)
optimal_cost = cost_matrix[r_opt, c_opt].sum()

print(f"\n✅ Phương án TỐI ƯU (Hungarian Algorithm):")
for i in range(4):
    print(f"   Công nhân {i} → Công việc {c_opt[i]} (chi phí: {cost_matrix[i, c_opt[i]]:.3f})")
print(f"   📊 TỔNG CHI PHÍ: {optimal_cost:.3f}")

print("\n" + "="*100)
print("🧠 GNN SẼ HỌC CÁCH TÌM PHƯƠNG ÁN NÀY NHƯ THẾ NÀO?")
print("="*100)

print("\n" + "▼"*50)
print("BƯỚC 1: CHUẨN BỊ INPUT (Biến Cost Matrix thành Graph)")
print("▼"*50)

cost_T = cost_matrix.T
x_input = np.concatenate((cost_matrix, cost_T), axis=0)

print("\n🔹 Ý tưởng: Biểu diễn bài toán dưới dạng đồ thị (graph)")
print("   - 4 công nhân → 4 nodes đầu tiên")
print("   - 4 công việc → 4 nodes tiếp theo (từ cost matrix lật ngang)")
print("   - Tổng cộng: 8 nodes, mỗi node có 4 features (chi phí)")

print("\n📊 Input x_input: Shape (8 nodes × 4 features)")
print("\n   CÔNG NHÂN (4 nodes đầu):")
for i in range(4):
    print(f"   Node {i} (CN{i}): {x_input[i]} ← Chi phí CN{i} cho 4 công việc")

print("\n   CÔNG VIỆC (4 nodes sau - từ cost matrix transpose):")
for i in range(4, 8):
    print(f"   Node {i} (CV{i-4}): {x_input[i]} ← Chi phí 4 công nhân cho CV{i-4}")

print("\n📈 Thống kê Input:")
print(f"   • Shape: {x_input.shape} (8 nodes, mỗi node 4 số)")
print(f"   • Giá trị nhỏ nhất: {x_input.min():.6f}")
print(f"   • Giá trị lớn nhất: {x_input.max():.6f}")
print(f"   • Trung bình: {x_input.mean():.6f} (giá trị ở giữa)")
print(f"   • Độ lệch chuẩn: {x_input.std():.6f} (độ phân tán của dữ liệu)")

print("\n💡 Giải thích các chỉ số:")
print("   • Min/Max: Khoảng giá trị của chi phí")
print("   • Mean (Trung bình): Giá trị trung tâm của dữ liệu")
print("   • Std (Độ lệch chuẩn): Đo độ phân tán")
print("     - Std cao: Dữ liệu trải rộng (khác biệt nhiều)")
print("     - Std thấp: Dữ liệu tập trung (giống nhau)")

x = torch.from_numpy(x_input).float()
G = Data(x, edge_index)

print(f"\n🔗 Kết nối (Edges): {edge_index.shape[1]} cạnh")
print("   → Mỗi công nhân kết nối với mọi công việc (bipartite graph)")

input("\n⏸️  Nhấn ENTER để tiếp tục sang Conv1...")

print("\n" + "▼"*50)
print("BƯỚC 2: CONV1 - EdgeConv Layer 1 (Học đặc trưng)")
print("▼"*50)

print("\n🔹 Mục đích: MỞ RỘNG thông tin từ 4 features → 32 features")
print("   - Input: 8 nodes × 4 features")
print("   - Output: 8 nodes × 32 features")

print("\n🧮 Cách hoạt động:")
print("   1. Với mỗi cặp nodes có kết nối (i,j):")
print("      → Ghép nối features: [x_i, x_j] = 8 số")
print("   2. Đưa qua MLP (Multi-Layer Perceptron):")
print("      → Linear(8 → 64) → ReLU → Linear(64 → 32)")
print("   3. Lấy max từ tất cả neighbors → 32 features cho mỗi node")

with torch.no_grad():
    x1 = model.conv1(x, edge_index)

print("\n📊 Output Conv1: Shape (8 nodes × 32 features)")
print(f"   • Giá trị nhỏ nhất: {x1.min().item():.6f}")
print(f"   • Giá trị lớn nhất: {x1.max().item():.6f}")
print(f"   • Trung bình: {x1.mean().item():.6f}")
print(f"   • Độ lệch chuẩn: {x1.std().item():.6f}")

print("\n💡 Ý nghĩa:")
print("   - Conv1 đã học được 32 đặc trưng ẩn (hidden features)")
print("   - Mỗi node giờ có 32 số thay vì 4 số ban đầu")
print("   - Các số này mã hóa thông tin về:")
print("     • Chi phí của chính node đó")
print("     • Chi phí của các nodes láng giềng")
print("     • Mối quan hệ giữa công nhân và công việc")

print("\n🔍 Ví dụ: Node 0 (Công nhân 0) sau Conv1:")
print(f"   {x1[0].numpy()[:8]}... (chỉ hiện 8/32 số đầu)")
print("   → Đây là vector đặc trưng đã học được!")

input("\n⏸️  Nhấn ENTER để tiếp tục sang Conv2...")

print("\n" + "▼"*50)
print("BƯỚC 3: CONV2 - EdgeConv Layer 2 (Tạo dự đoán)")
print("▼"*50)

print("\n🔹 Mục đích: THU HẸP từ 32 features → 4 features (4 công việc)")
print("   - Input: 8 nodes × 32 features")
print("   - Output: 8 nodes × 4 features (logits)")

print("\n🧮 Cách hoạt động:")
print("   1. Ghép nối features của các cặp nodes: [x_i, x_j] = 64 số")
print("   2. Đưa qua MLP:")
print("      → Linear(64 → 64) → ReLU → Linear(64 → 4)")
print("   3. Max aggregation → 4 logits cho mỗi node")

with torch.no_grad():
    x2 = model.conv2(x1, edge_index)

print("\n📊 Output Conv2: Shape (8 nodes × 4 logits)")

print("\n   CÔNG NHÂN (4 nodes đầu) - Dự đoán nên chọn công việc nào:")
for i in range(4):
    print(f"   Node {i} (CN{i}): {x2[i].numpy()}")
    print(f"      → Logit cao nhất ở vị trí {x2[i].argmax().item()} (CV{x2[i].argmax().item()})")

print("\n   CÔNG VIỆC (4 nodes sau) - Dự đoán nên được công nhân nào làm:")
for i in range(4, 8):
    print(f"   Node {i} (CV{i-4}): {x2[i].numpy()}")

print("\n📈 Thống kê Conv2:")
print(f"   • Min: {x2.min().item():.6f}")
print(f"   • Max: {x2.max().item():.6f}")
print(f"   • Mean: {x2.mean().item():.6f}")
print(f"   • Std: {x2.std().item():.6f}")

print("\n💡 Ý nghĩa các chỉ số:")
print("   - Giá trị CÀO NHẤT trong mỗi hàng → công việc được ƯU TIÊN")
print("   - Std cao → model TỰ TIN (có sự phân biệt rõ ràng)")
print("   - Std thấp → model KHÔNG CHẮC (các giá trị gần bằng nhau)")

input("\n⏸️  Nhấn ENTER để tiếp tục sang Readout...")

print("\n" + "▼"*50)
print("BƯỚC 4: READOUT - Tổng hợp cuối cùng")
print("▼"*50)

print("\n🔹 Mục đích: Tạo ma trận dự đoán 4×4")
print("   - Input: 8 nodes × 4 features")
print("   - Output: 4 công nhân × 4 công việc")

print("\n🧮 Cách hoạt động:")
print("   1. Transpose: 8×4 → 4×8")
print("   2. Linear layer: 4×8 → 4×4")
print("   3. Mỗi hàng = sở thích của 1 công nhân cho 4 công việc")

with torch.no_grad():
    output = model.readout(x2.T)

print("\n📊 Output cuối cùng: Shape (4 công nhân × 4 công việc)")
print("\n🔥 MA TRẬN LOGITS (số càng cao = càng thích):")
print()
for i in range(4):
    print(f"   CN{i}: ", end="")
    for j in range(4):
        val = output[i,j].item()
        if val == output[i].max().item():
            print(f"[CV{j}: {val:7.3f}]★ ", end="")  # Đánh dấu max
        else:
            print(f"[CV{j}: {val:7.3f}]  ", end="")
    print(f" → Thích nhất: CV{output[i].argmax().item()}")

print("\n📈 Thống kê mỗi hàng (công nhân):")
for i in range(4):
    print(f"   CN{i}: Min={output[i].min().item():7.3f}, Max={output[i].max().item():7.3f}, "
          f"Mean={output[i].mean().item():7.3f}, Std={output[i].std().item():.3f}")

print("\n💡 Ý nghĩa Std (Độ lệch chuẩn) cho mỗi hàng:")
print("   - Std CAO (>5): Model rất TỰ TIN, có lựa chọn RÕ RÀNG")
print("   - Std TRUNG BÌNH (2-5): Model khá chắc chắn")
print("   - Std THẤP (<2): Model KHÔNG CHẮC, khó quyết định")

input("\n⏸️  Nhấn ENTER để xem Softmax (chuyển thành xác suất)...")

print("\n" + "▼"*50)
print("BONUS: SOFTMAX - Chuyển Logits thành Xác suất (%)")
print("▼"*50)

with torch.no_grad():
    probs = F.softmax(output, dim=1)

print("\n🎲 MA TRẬN XÁC SUẤT (mỗi hàng tổng = 100%):")
print()
for i in range(4):
    print(f"   CN{i}: ", end="")
    for j in range(4):
        prob = probs[i,j].item() * 100
        if prob > 50:
            print(f"[CV{j}: {prob:5.1f}%]★ ", end="")
        else:
            print(f"[CV{j}: {prob:5.1f}%]  ", end="")
    print(f" → Tổng: {probs[i].sum().item()*100:.1f}%")

print("\n💡 Ý nghĩa:")
print("   - Softmax biến logits thành xác suất (0-100%)")
print("   - Xác suất càng cao = càng nên chọn")
print("   - VD: CN0 chọn CV1 với xác suất 98.6%")

input("\n⏸️  Nhấn ENTER để xem Greedy Algorithm...")

print("\n" + "▼"*50)
print("BƯỚC 5: GREEDY ALGORITHM - Tránh xung đột")
print("▼"*50)

print("\n🔹 Vấn đề: Mỗi công nhân chỉ chọn công việc YÊU THÍCH NHẤT")
print("   → Có thể nhiều người cùng chọn 1 công việc!")
print("   → Cần thuật toán để phân công KHÔNG TRÙNG")

print("\n🧮 Greedy Algorithm hoạt động:")
print("   1. Tìm giá trị CAO NHẤT trong toàn bộ ma trận")
print("   2. Phân công: công nhân đó → công việc đó")
print("   3. LOẠI BỎ hàng và cột đã chọn (mask = giá trị rất nhỏ)")
print("   4. Lặp lại cho đến hết")

with torch.no_grad():
    pred = avoid_coll(output.numpy(), param_dict)

print(f"\n✅ KẾT QUẢ sau Greedy:")
for i in range(4):
    print(f"   Công nhân {i} → Công việc {pred[i]} (chi phí: {cost_matrix[i, pred[i]]:.3f})")

gnn_cost = sum(cost_matrix[i, pred[i]] for i in range(4))
print(f"\n📊 TỔNG CHI PHÍ của GNN: {gnn_cost:.3f}")
print(f"📊 TỔNG CHI PHÍ tối ưu:  {optimal_cost:.3f}")

if np.array_equal(pred, c_opt):
    print("\n🎉 HOÀN HẢO! GNN tìm ra được phương án TỐI ƯU!")
else:
    print(f"\n⚠️  GNN không tìm ra tối ưu (sai lệch: {gnn_cost - optimal_cost:.3f})")

print("\n" + "="*100)
print("📚 TÓM TẮT CÁC CHỈ SỐ THỐNG KÊ")
print("="*100)

print("""
1️⃣  SHAPE (Kích thước):
   - (8, 4) = 8 hàng, 4 cột
   - (4, 32) = 4 hàng, 32 cột
   
2️⃣  MIN (Giá trị nhỏ nhất):
   - Số nhỏ nhất trong ma trận
   - Quan trọng để biết phạm vi dữ liệu
   
3️⃣  MAX (Giá trị lớn nhất):
   - Số lớn nhất trong ma trận
   - Logit cao = được ưu tiên chọn
   
4️⃣  MEAN (Trung bình):
   - Tổng tất cả / số phần tử
   - Giá trị trung tâm của dữ liệu
   - Cho biết dữ liệu nghiêng về đâu
   
5️⃣  STD (Độ lệch chuẩn):
   - Đo độ PHÂN TÁN của dữ liệu
   - Std CAO: Dữ liệu rải rộng, khác biệt lớn
   - Std THẤP: Dữ liệu tập trung, gần bằng nhau
   - Trong GNN:
     • Std cao = model TỰ TIN
     • Std thấp = model KHÔNG CHẮC CHẮN
""")

print("\n" + "="*100)
print("🎯 VÍ DỤ DỄ HIỂU VỀ STD")
print("="*100)

print("""
Giả sử 2 công nhân đánh giá sở thích (0-10):

Công nhân A: [2, 9, 3, 2]
  → Rõ ràng THÍCH công việc 2 (9 điểm) nhất!
  → Std = 3.5 (CAO) → TỰ TIN

Công nhân B: [5, 6, 5, 5]
  → Không rõ thích cái nào, gần bằng nhau
  → Std = 0.5 (THẤP) → KHÔNG CHẮC

➡️  Trong GNN, Std cao = model có sự lựa chọn RÕ RÀNG
""")

print("\n" + "="*100)
print("✅ HOÀN TẤT!")
print("="*100)
