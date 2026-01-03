# GIẢI THÍCH GNN CHO BÀI TOÁN LINEAR ASSIGNMENT - TIẾNG VIỆT

## 🎯 Bài toán

**Phân công 4 công nhân cho 4 công việc sao cho TỔNG CHI PHÍ THẤP NHẤT**

Ví dụ:
```
Cost Matrix:
         CV0    CV1    CV2    CV3
CN0:    0.77   0.56   0.54   0.66
CN1:    0.57   0.66   0.36   0.91
CN2:    0.71   0.91   0.26   0.54
CN3:    0.31   0.52   0.96   0.53

Phương án tối ưu:
- Công nhân 0 → Công việc 1 (0.56)
- Công nhân 1 → Công việc 2 (0.36)
- Công nhân 2 → Công việc 3 (0.54)
- Công nhân 3 → Công việc 0 (0.31)
TỔNG: 1.78
```

---

## 🧠 Cách GNN hoạt động

### Bước 1: Chuẩn bị Input (8 nodes × 4 features)

```
Input = [Cost Matrix; Cost Matrix Transpose]
      = [4 công nhân; 4 công việc]
      = 8 nodes, mỗi node 4 features
```

**Ví dụ:**
- Node 0 (CN0): `[0.77, 0.56, 0.54, 0.66]` - chi phí CN0 cho 4 CV
- Node 4 (CV0): `[0.77, 0.57, 0.71, 0.31]` - chi phí 4 CN cho CV0

**Tạo đồ thị bipartite:** Mỗi công nhân kết nối với mọi công việc

---

### Bước 2: Conv1 - Mở rộng features (8×4 → 8×32)

```
EdgeConv Layer:
  Input:  8 nodes × 4 features
  Output: 8 nodes × 32 features
  
Cách hoạt động:
1. Ghép nối features của nodes láng giềng: [x_i, x_j]
2. Đưa qua MLP: Linear(8→64) → ReLU → Linear(64→32)
3. Max aggregation
```

**Mục đích:** Học các đặc trưng ẩn (hidden features) từ quan hệ giữa công nhân và công việc

---

### Bước 3: Conv2 - Thu hẹp về predictions (8×32 → 8×4)

```
EdgeConv Layer:
  Input:  8 nodes × 32 features  
  Output: 8 nodes × 4 logits
  
Cách hoạt động:
1. Ghép nối features: [x_i, x_j]
2. Đưa qua MLP: Linear(64→64) → ReLU → Linear(64→4)
3. Max aggregation
```

**Mục đích:** Tạo dự đoán ban đầu - mỗi node có 4 logits (preference scores)

**Ví dụ Output:**
```
CN0: [1.84, -0.91, 1.03, 0.82]  → Thích CV0 nhất
CN1: [-0.24, 0.26, -1.90, 2.71] → Thích CV3 nhất
CN2: [1.96, 3.06, -1.03, 0.69] → Thích CV1 nhất
CN3: [-1.21, 1.60, 2.85, 1.51] → Thích CV2 nhất
```

---

### Bước 4: Readout - Tổng hợp cuối cùng (8×4 → 4×4)

```
Linear Layer:
  Input:  Transpose(8×4) = 4×8
  Output: 4×4 matrix
  
Cách hoạt động:
  Linear(8 → 4) projection
```

**Output cuối cùng: Ma trận 4×4 logits**

```
         CV0     CV1     CV2     CV3
CN0:   -6.63    6.45   -7.16    2.17   ← Thích CV1 (6.45)
CN1:    2.69    1.48    6.32   -2.60   ← Thích CV2 (6.32)
CN2:   -4.18   -6.13    4.30    5.69   ← Thích CV3 (5.69)
CN3:    5.89   -1.31  -11.47    1.63   ← Thích CV0 (5.89)
```

**Lưu ý:** Đây là **RAW LOGITS** (không phải xác suất)
- Có thể âm
- KHÔNG tổng = 1
- Càng cao = càng thích

---

### Bước 5: Softmax - Chuyển thành xác suất (optional)

```
Áp dụng softmax cho mỗi hàng:
         CV0     CV1     CV2     CV3
CN0:    0.0%   98.6%    0.0%    1.4%   ← 98.6% chắc chắn chọn CV1
CN1:    2.6%    0.8%   96.7%    0.0%   ← 96.7% chắc chắn chọn CV2
CN2:    0.0%    0.0%   20.0%   80.0%   ← 80.0% chọn CV3
CN3:   98.5%    0.1%    0.0%    1.4%   ← 98.5% chắc chắn chọn CV0
```

**Mỗi hàng tổng = 100%** ✅

---

### Bước 6: Greedy Algorithm - Tránh xung đột

**Vấn đề:** Nếu mỗi công nhân chỉ chọn công việc yêu thích nhất:
```
CN0 → CV1
CN1 → CV2  
CN2 → CV3
CN3 → CV0
```

Có thể bị **TRÙNG** (nhiều người chọn cùng 1 công việc)!

**Giải pháp - Greedy Algorithm:**

```
1. Tìm giá trị MAX trong toàn bộ ma trận
2. Phân công: công nhân đó → công việc đó
3. MASK (loại bỏ) hàng và cột đã chọn
4. Lặp lại cho đến hết
```

**Ví dụ:**
```
Step 1: Max = 6.45 tại (CN0, CV1)
  → Phân công: CN0 → CV1
  → Mask hàng 0 và cột 1
  
Step 2: Max = 6.32 tại (CN1, CV2)
  → Phân công: CN1 → CV2
  → Mask hàng 1 và cột 2
  
Step 3: Max = 5.89 tại (CN3, CV0)
  → Phân công: CN3 → CV0
  → Mask hàng 3 và cột 0
  
Step 4: Max = 5.69 tại (CN2, CV3)
  → Phân công: CN2 → CV3
  → Mask hàng 2 và cột 3
```

**Kết quả:** `[1, 2, 3, 0]` - Mỗi công nhân 1 công việc, không trùng! ✅

---

## 📊 Giải thích các CHỈ SỐ THỐNG KÊ

### 1. Shape (Kích thước)
```
Shape: (8, 4) = 8 hàng, 4 cột
Shape: (4, 32) = 4 hàng, 32 cột
```

### 2. Min (Giá trị nhỏ nhất)
```
Min: -11.47
→ Số nhỏ nhất trong ma trận
→ Cho biết cận dưới của dữ liệu
```

### 3. Max (Giá trị lớn nhất)
```
Max: 6.45
→ Số lớn nhất trong ma trận
→ Logit cao = được ưu tiên chọn
```

### 4. Mean (Trung bình)
```
Mean: 0.61
Công thức: Tổng tất cả / số phần tử
→ Giá trị trung tâm
→ Cho biết dữ liệu nghiêng về đâu
```

### 5. Std (Độ lệch chuẩn) ⭐ QUAN TRỌNG

**Đo độ PHÂN TÁN của dữ liệu:**

```
Std CAO (>5):
  → Dữ liệu rải rộng, khác biệt lớn
  → Model TỰ TIN, có lựa chọn RÕ RÀNG
  
Std THẤP (<2):
  → Dữ liệu tập trung, gần bằng nhau
  → Model KHÔNG CHẮC CHẮN, khó quyết định
```

**Ví dụ dễ hiểu:**

```
Công nhân A đánh giá 4 công việc:
[2, 9, 3, 2]
  → Rõ ràng THÍCH công việc 2 (9 điểm) nhất!
  → Std = 3.5 (CAO)
  → Model TỰ TIN ✅

Công nhân B đánh giá 4 công việc:
[5, 6, 5, 5]
  → Không rõ thích cái nào, gần bằng nhau
  → Std = 0.5 (THẤP)
  → Model KHÔNG CHẮC ⚠️
```

---

## ⚠️ Khi nào Greedy THẤT BẠI?

**Tình huống:** Sau 3 bước greedy, tất cả giá trị còn lại **BẰNG NHAU**

```
Ví dụ Sample #214:

Initial logits: Có variance tốt ✓
[-10.67  -1.60   0.02  -2.63]
[  2.04   2.29  -7.51  -2.16]
[  1.71  -9.20  -3.39   2.63]
[-11.53   1.74  -0.18   0.33]

Step 1: Pick (2,3) ✓
Step 2: Pick (1,1) ✓  
Step 3: Pick (0,2) ✓

Step 4: ALL remaining values = -11.53 ❌
  → Toàn bộ còn lại uniform!
  → np.where() returns ALL 16 positions
  → pp matrix becomes all 1s
  → argmax returns [0,0,0,0] (INVALID!)
```

**Tỷ lệ thất bại:** 75/20,000 = 0.38%

---

## 🎯 Tổng kết

### Luồng xử lý
```
Cost Matrix (4×4)
    ↓
Input Prep (8×4)     ← Biến thành graph
    ↓
Conv1 (8×32)         ← Mở rộng features
    ↓
Conv2 (8×4)          ← Thu hẹp thành logits
    ↓
Readout (4×4)        ← Tổng hợp cuối cùng
    ↓
Softmax (4×4)        ← Chuyển thành xác suất (optional)
    ↓
Greedy (4,)          ← Tránh xung đột
    ↓
Assignment [1,2,3,0] ← Kết quả cuối cùng
```

### Kết quả
- ✅ **92.02% full row accuracy** - Tìm đúng phương án tối ưu
- ✅ **99.62% validity** - Greedy tạo phân công hợp lệ
- ⚠️ **0.38% failure** - Greedy thất bại khi còn lại uniform

### Ưu điểm
- Nhanh hơn Hungarian Algorithm (~180x)
- Học được patterns từ dữ liệu
- Khả năng generalize tốt

### Nhược điểm
- Không đảm bảo 100% tối ưu
- Greedy có thể fail trong rare cases
- Phụ thuộc vào quality của training data

---

## 💡 Key Takeaways

1. **GNN không output xác suất trực tiếp**, mà output **raw logits**
2. **CrossEntropyLoss** tự động apply softmax trong quá trình training
3. **Greedy algorithm** hoạt động trên raw logits (không phải probabilities)
4. **Std cao** = model tự tin, có sự lựa chọn rõ ràng
5. **Std thấp** = model không chắc chắn, các giá trị gần nhau
6. Greedy thất bại khi remaining values uniform sau masking
