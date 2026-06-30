# So sánh RDLinear vs DLinear

## Khác biệt cốt lõi

`RDLinear` = `DLinear` + **RevIN (Reversible Instance Normalization)**. Kiến trúc decomposition + linear bên trong là **giống hệt nhau**, chỉ thêm lớp chuẩn hóa / giải chuẩn hóa ở đầu vào và đầu ra.

| Khía cạnh | DLinear | RDLinear |
|---|---|---|
| `kernel_size` | Hard-coded `25` | `getattr(configs, 'moving_avg', 25)` — cấu hình được |
| Chuẩn hóa input | Không | Trừ mean (và chia std nếu `revin_mode='std'`) trên chiều thời gian, **per-instance, per-channel** |
| Affine học được | Không | Tùy chọn `revin_affine` → `affine_weight`, `affine_bias` |
| De-normalize output | Không | Cộng lại mean (và nhân std), đảo affine |
| Tham số phụ | 0 | `2 * channels` nếu bật affine; còn lại chỉ là statistic runtime (`.detach()`) |

Các block `moving_avg`, `series_decomp`, `Linear_Seasonal`, `Linear_Trend` (cả `individual` lẫn shared) — định danh.

## RevIN làm gì về mặt toán học

Với input $x \in \mathbb{R}^{B \times L \times C}$:

$$
\mu_c = \frac{1}{L}\sum_{t=1}^{L} x_{t,c}, \quad
\sigma_c = \sqrt{\frac{1}{L}\sum_{t=1}^{L}(x_{t,c}-\mu_c)^2 + \epsilon}
$$

Forward: $\tilde{x} = (x - \mu)/\sigma$, model dự báo $\tilde{y}$, output $y = \tilde{y}\cdot\sigma + \mu$.

Vì `means` / `stdev` được `.detach()`, gradient không chảy qua phần thống kê → mạng học trong không gian đã chuẩn hóa, ổn định hơn.

## Có cải thiện performance không?

**Có, trong phần lớn các trường hợp**, đặc biệt:

1. **Distribution shift** giữa train/val/test (rất phổ biến trên ETTh, ETTm, Electricity, Traffic, Weather). RevIN khử mean/scale của từng look-back window nên dự báo bám theo level hiện tại thay vì level "trung bình của tập train". Trong paper RevIN (Kim et al., ICLR 2022) và các benchmark sau đó (PatchTST, iTransformer), RevIN thường giảm MSE **~5–20%**, mạnh nhất ở các chuỗi non-stationary và horizon dài (336 / 720).
2. **NLinear** thực chất là DLinear-không-decomp + "subtract last value" — một dạng instance normalization đơn giản. Việc nó thường thắng DLinear trên benchmark gốc của repo này chính là bằng chứng rằng thêm normalization có ích. `revin_mode='mean'` về cơ bản tổng quát hóa ý tưởng đó (trừ mean toàn cửa sổ thay vì giá trị cuối) + giữ decomposition.

**Khi nào không (hoặc ít) cải thiện:**

- Chuỗi đã stationary sẵn (đã được preprocess scale toàn cục tốt) — RevIN có thể trung tính hoặc hơi xấu hơn vì làm mất thông tin level tuyệt đối.
- Horizon rất ngắn với chuỗi mượt — biên độ cải thiện nhỏ.
- Bật `revin_affine=True` trên dataset nhỏ đôi khi gây overfit nhẹ; thường `affine=False` an toàn hơn, hoặc chỉ bật khi `enc_in` nhỏ.

## Lưu ý nhỏ về code

- `_denormalize` dùng `self.affine_weight + self.revin_eps` để tránh chia 0, hợp lý; nhưng vì `affine_weight` init = 1, một số implementation chuẩn của RevIN dùng `(affine_weight**2 + eps)` hoặc `affine_weight + eps_signed`. Hiện tại ổn miễn weight không tiến về 0 âm.
- Khi `revin_mode='mean'`, `stdev` là `None` và `_denormalize` bỏ qua đúng — không bug.
- `means.detach()` đảm bảo RevIN không "rò" gradient bậc hai qua thống kê, đúng spec gốc.

## Khuyến nghị thực nghiệm

Chạy ablation trên cùng dataset / horizon với:

1. DLinear baseline
2. RDLinear `revin_mode='mean'`, `affine=False`
3. RDLinear `revin_mode='std'`, `affine=False`
4. RDLinear `revin_mode='std'`, `affine=True`

Kỳ vọng: (3) ≥ (2) > (1) trên ETTh1/2, Electricity, Weather ở `pred_len ∈ {96, 192, 336, 720}`. (4) cải thiện thêm tùy dataset.
