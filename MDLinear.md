# So sánh MDLinear vs RDLinear

## Khác biệt cốt lõi

`MDLinear` = `RDLinear` + **multi-scale decomposition**. Cả hai đều bọc RevIN ở đầu vào / đầu ra và đều dùng cùng một head linear seasonal + trend. Điểm khác **duy nhất** nằm ở khối tách trend: RDLinear chỉ dùng *một* moving average kernel cố định, trong khi MDLinear học một **hỗn hợp có softmax-gate trên nhiều kernel sizes** (FEDformer-style trend mixture).

| Khía cạnh | RDLinear | MDLinear |
|---|---|---|
| Khối tách trend | `series_decomp(kernel_size)` — 1 kernel | `series_decomp_multi(kernel_sizes)` — K kernel |
| Cấu hình kernel | `--moving_avg` (1 số) | `--decomp_kernels` (danh sách, fallback về `--moving_avg`) |
| Trend = | `MA_k(x)` cố định | $\sum_{i=1}^{K} w_{i}(x) \cdot \text{MA}_{k_i}(x)$, với $w$ là softmax do `nn.Linear(1, K)` sinh ra trên từng `(batch, time, channel)` |
| Tham số phụ vs RDLinear | – | `2K` (weight + bias của `gate`); các `AvgPool1d` không có tham số học |
| RevIN | Bật mặc định | Bật mặc định (không còn flag `use_revin` — luôn áp dụng) |
| Affine RevIN | `revin_affine` | `revin_affine` (giống nhau) |
| Linear_Seasonal / Linear_Trend | Giống hệt | Giống hệt |
| Padding cho moving_avg | Đối xứng `(k-1)//2` mỗi bên | Đối xứng `(k-1)//2` trước, `k-1-(k-1)//2` sau (xử lý cả `k` chẵn nếu lỡ truyền vào) |
| Bảo đảm kernel lẻ | Không bắt buộc | Tự bump `k → k+1` nếu chẵn |

Các block `moving_avg`, `Linear_Seasonal`, `Linear_Trend`, `_normalize`, `_denormalize` — định danh.

## Multi-scale decomposition làm gì về mặt toán học

Với input $x \in \mathbb{R}^{B \times L \times C}$ và danh sách kernel sizes $\{k_1, \dots, k_K\}$:

1. Tính K trend ứng viên: $T_i = \text{MA}_{k_i}(x) \in \mathbb{R}^{B \times L \times C}$.
2. Sinh logits gating từ chính giá trị input: $z = W \cdot x_{\text{unsq}} + b$, với $W \in \mathbb{R}^{K \times 1}, b \in \mathbb{R}^{K}$ — tức gate quyết định theo **giá trị tại mỗi (b, t, c)**, không chia sẻ trên cửa sổ.
3. Trộn bằng softmax theo chiều kernel:

$$
\text{trend}_{b,t,c} = \sum_{i=1}^{K} \text{softmax}_i\bigl(z_{b,t,c}\bigr)\,T_{i,b,t,c}
$$

4. Phần seasonal = $x - \text{trend}$, hai nhánh đi qua linear riêng như DLinear/RDLinear, sau đó cộng lại và de-normalize.

Vì gate chỉ phụ thuộc giá trị tức thời (`x.unsqueeze(-1)`), mỗi vị trí có thể chọn một thang trend khác nhau — vùng có biến động cao có thể dồn trọng số sang kernel ngắn (`k=9, 13`) để bám sát; vùng phẳng có thể chọn kernel dài (`k=49`) để bỏ qua nhiễu.

## Có cải thiện performance không?

**Có khả năng cải thiện**, nhưng **không đảm bảo**. Một single kernel `k=25` là một inductive bias rất mạnh và thường đã đủ tốt; MDLinear chỉ thắng khi bias đó thực sự sai cho dataset.

**Khi nào MDLinear vượt RDLinear:**

1. **Đa chu kỳ rõ rệt** — Traffic, Electricity có cả mẫu daily và weekly. Một kernel `25` (~1 ngày dữ liệu hourly) bỏ sót cấu trúc tuần; mixture với `{13, 25, 49}` hoặc `{25, 97, 169}` cho gate cơ hội chọn scale phù hợp theo vùng.
2. **Regime switching** — Exchange-Rate, một phần ETTh2: chuỗi xen kẽ đoạn ổn định và đoạn biến động. Gate học được điều này nhờ depend on input value.
3. **Horizon dài (336 / 720)** — sai số trend tích lũy mạnh, decomposition chính xác hơn được hưởng lợi nhiều hơn.

**Khi nào MDLinear KHÔNG cải thiện (hoặc tệ hơn) RDLinear:**

- Dataset có chu kỳ duy nhất, đã được kernel `25` xử lý tốt (ETTm1/m2 nhiều cấu hình) — mixture chỉ thêm tham số nhưng không thêm tín hiệu.
- Dataset nhỏ (ILI ~ vài trăm samples) — gate `nn.Linear(1, K)` tuy nhỏ nhưng vẫn dễ overfit và làm trend nhiễu hơn.
- Khi `--decomp_kernels` chứa kernel quá lớn so với `seq_len` (ví dụ `k=97` với `seq_len=104`): padding edge-repeat sẽ trải dài đầu/cuối và bóp méo trend.

## Lưu ý nhỏ về code

- `self.moving_avgs = nn.ModuleList([...])` — đúng pattern để `.to(device)` / `.train()` lan tỏa. Bản gốc của FEDformer dùng Python list cho `moving_avg`, sẽ không di chuyển cùng model (đã sửa ở đây).
- `gate` được áp dụng trên `x.unsqueeze(-1)` → shape `[B, L, C, 1]` → `[B, L, C, K]`. Có nghĩa **mỗi channel dùng chung tham số gate**, nhưng có gate value riêng tại mỗi (b, t, c). Nếu muốn gate riêng cho từng channel có thể đổi thành `nn.Linear(C, K)` hoặc per-channel `ModuleList`, đổi lại sẽ tốn `K*C` tham số.
- Bump kernel chẵn → lẻ: `[k if k % 2 == 1 else k + 1 ...]` để giữ padding đối xứng. Lưu ý biến đã bump được gán vào `self.kernel_sizes`, nhưng `series_decomp_multi(kernel_sizes)` ở dòng dưới đang nhận **biến gốc chưa bump**. Trên đường padding hiện tại của `moving_avg`, kernel chẵn vẫn chạy được (padding sau = `k-1-(k-1)//2`), nên không lỗi nhưng có thể không khớp với `self.kernel_sizes` dùng để log. Nếu cần nhất quán: truyền `self.kernel_sizes` vào `series_decomp_multi`.
- RevIN luôn được áp dụng (không có gate `use_revin`); nếu muốn ablation "MDLinear không RevIN" cần thêm cờ lại như DLinear, hoặc tạm thời tạo biến thể bằng cách gọi thẳng `series_decomp_multi` từ DLinear.
- `means.detach()` / `stdev.detach()` — giữ nguyên hành vi RevIN chuẩn, gradient không chảy qua thống kê.

## Khuyến nghị thực nghiệm

Chạy 4 cấu hình trên cùng (dataset, `seq_len`, `pred_len`, seed):

1. **DLinear** — baseline (1 kernel, không RevIN).
2. **RDLinear** `revin_mode='std'`, `affine=False` — đo riêng đóng góp RevIN.
3. **MDLinear** `--decomp_kernels 13 25 49` — đo riêng đóng góp multi-scale (vì cả 2/3 đều có RevIN).
4. **MDLinear** `--decomp_kernels` tuỳ dataset — Traffic/Electricity; ETT* thử `{9, 25, 49}`; ILI thử `{9 13 25}` (nhỏ vì `seq_len=104`).

Phân tích cặp (2) → (3): chênh lệch MSE/MAE là đóng góp thuần của multi-scale decomposition. Cặp (3) → (4): hưởng lợi từ việc tune danh sách kernel theo dataset.

Kỳ vọng tổng quát:

- Traffic, Electricity, Weather (`pred_len ∈ {336, 720}`): (4) ≥ (3) ≥ (2) > (1).
- ETTm1/m2, ILI: (3) ≈ (2), tức multi-scale có thể không thêm gì; nếu vậy giữ RDLinear cho gọn.
- Exchange-Rate: biến động rất lớn — kết quả thường phụ thuộc seed; cần `--itr ≥ 3` để kết luận.
