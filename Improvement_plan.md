# Improvement Plan for DLinear

A survey of approaches to improve **DLinear** for long-term time series forecasting (LTSF),
prioritizing **forecast accuracy** (lower MSE/MAE) on the standard benchmarks. Grounded in the
paper *"Are Transformers Effective for Time Series Forecasting?"* (arXiv:2205.13504v3) and the
implementation in this repository.

---

## 1. Background

DLinear ([models/DLinear.py](models/DLinear.py)) decomposes each input series into a **trend**
component (moving average, `series_decomp`) and a **seasonal/remainder** component
(`input − trend`), feeds each through a single linear layer mapping `seq_len → pred_len`, and
sums the two outputs. With `--individual` it learns a separate linear pair per channel; otherwise
weights are shared across channels. It does **direct multi-step (DMS)** forecasting, is tiny
(~140K params), interpretable, and — unlike Transformers — its accuracy *improves* with longer
look-back windows.

The paper itself flags DLinear's limitations, which define the opportunity space:

- **Limited capacity** — a single linear layer cannot capture temporal dynamics around
  **change points** or regime shifts.
- **No cross-variate modeling** — channels are forecast independently; inter-series correlations
  (useful on Traffic/Electricity) are ignored.
- **Fixed decomposition** — a single hardcoded moving-average kernel (`kernel_size = 25`).
- **No distribution-shift handling** — that is NLinear's trick, not built into DLinear.

Two implementation facts make experimentation cheap:

- The forward path is selected by `'Linear' in args.model` in
  [exp/exp_main.py](exp/exp_main.py#L70), so **any new model whose name contains "Linear"** is
  driven with the single-input `self.model(batch_x)` call. A new variant needs only a
  `models/<Name>.py` exposing a `Model(configs)` class plus one entry in the `model_dict` at
  [exp/exp_main.py:35](exp/exp_main.py#L35) — no training-loop edits.
- Existing run scripts under [scripts/EXP-LongForecasting/Linear/](scripts/EXP-LongForecasting/Linear/)
  can be reused for any variant by swapping `--model`.

---

## 2. Ranked Improvement Directions

Ranked by expected **accuracy gain per unit of effort**. Each entry: *idea → why it helps →
cost/complexity → where to implement → risk*.

### A. Reversible Instance Normalization (RevIN) — *highest priority*
- **Idea:** Before decomposition, normalize each input instance per-channel (subtract mean,
  divide by std over the look-back), forecast in normalized space, then de-normalize the output
  (with optional learnable affine `γ, β`). This generalizes NLinear's "subtract last value" trick.
- **Why it helps:** Directly attacks distribution shift between look-back and horizon — the single
  biggest source of error on ETTh2, Weather, and Exchange-Rate. Combining it with DLinear's
  decomposition gets *both* the trend/seasonal split and shift-robustness.
- **Cost:** Low. ~30 lines; negligible extra parameters.
- **Implement:** New `models/RDLinear.py` (copy DLinear, wrap `forward` with normalize/denormalize;
  store per-instance mean/std). Add to `model_dict`.
- **Risk:** Low. Std normalization can mildly hurt already-stationary, strongly-seasonal series
  (Traffic/Electricity); keep a flag to fall back to mean-only (NLinear-style) subtraction.

---

### Architecture-focused improvements for RDLinear

Every direction below is a **replacement** of a part of RDLinear's architecture, not a
parallel branch bolted on top. Prior attempts that *added* capacity next to the linear
predictor (multi-branch MLPs, cross-variate mixers) overfit on this benchmark; the directions
here instead restructure the predictor itself, and most of them **shrink** the parameter count
(`seq_len·pred_len` per branch is already the dominant term). All can be built on top of
[models/RDLinear.py](models/RDLinear.py) — keep RevIN and `series_decomp` unchanged.

### F. Low-rank / bottleneck linear predictor
- **Idea:** Factor each `Linear(seq_len → pred_len)` (seasonal and trend) as
  `Linear(seq_len → r) → Linear(r → pred_len)` with a small rank `r ∈ {16, 32, 64}`. Optionally
  add a GELU between the two, but the plain linear factorization is the safer starting point.
- **Why it helps:** Parameter count drops from `seq_len·pred_len` to `(seq_len + pred_len)·r`.
  At `seq_len=336, pred_len=720, r=32` that is 34K vs 242K — an ~86 % reduction, i.e. a strong
  built-in regularizer. The bottleneck also acts as a learned low-rank summary of the look-back,
  which lines up with the fact that DLinear's learned linear weights are empirically low-rank.
- **Cost:** ~5 lines per branch in [models/RDLinear.py](models/RDLinear.py); add a `--lr_rank` arg.
- **Risk:** Low. Sweep `r`; if too small on Traffic/Electricity, MSE plateaus rather than blows up.

### G. Frequency-domain (RFFT) predictor
- **Idea:** Route the seasonal residual through the frequency domain. `torch.fft.rfft` the
  normalized input, keep the lowest `K` complex bins (learnable amplitude + phase, or a full
  complex linear map on those bins), zero-pad to `pred_len // 2 + 1`, then `irfft` to obtain
  the forecast. Trend branch stays a plain linear layer.
- **Why it helps:** Encodes a **band-limited periodicity prior** directly into the architecture,
  which is the correct inductive bias for the seasonal component. Learnable parameters are only
  `O(K)` complex weights per branch — often an order of magnitude below the linear baseline. It
  is also the linear-model analogue of FEDformer / FreTS / FITS.
- **Cost:** ~30 lines using `torch.fft.rfft` / `torch.fft.irfft`.
- **Risk:** Low–moderate. Aperiodic series (Exchange, ILI) lose some high-frequency detail if
  `K` is set too aggressively; expose `K` as `--freq_k`.

### H. Fourier basis-expansion output (N-BEATS-style seasonal head)
- **Idea:** Instead of predicting `pred_len` samples directly, have the seasonal linear layer
  output `2·B` coefficients for a small set of `B` sine/cosine basis functions defined on
  `[0, pred_len)`. The forecast is a fixed matrix multiply `Ŷ = Φ · c`, where
  `Φ ∈ ℝ^{pred_len × 2B}` is a precomputed sin/cos basis (register as a buffer) and
  `c ∈ ℝ^{2B}` comes from the linear layer.
- **Why it helps:** Even stronger periodicity / smoothness prior than G — the output space is
  hard-constrained to be a low-order Fourier signal. Trainable parameters shrink from
  `seq_len·pred_len` to `seq_len·2B` with `B ≪ pred_len / 2`. Trend branch either stays linear
  or gets a polynomial basis of degree 2–3 (N-BEATS' trend block).
- **Cost:** ~20 lines + one `register_buffer('basis', ...)`.
- **Risk:** Low; under-parameterization on fast-changing / non-periodic series is possible, but
  the trend branch absorbs the smooth non-periodic component.

### I. Patch-based linear projection
- **Idea:** Split the look-back of length `seq_len` into non-overlapping patches of length `P`
  (e.g. 8, 16, 24). Reduce each patch to a single value (average, or a learned `Linear(P → 1)`)
  to produce `⌈seq_len / P⌉` tokens. Apply a linear map from those tokens to `pred_len`
  (or to `⌈pred_len / P⌉` output patches followed by nearest-neighbour upsample). PatchTST-
  style, but linear and per-channel.
- **Why it helps:** Divides the effective input dimension by `P`, so parameters drop by roughly
  the same factor. Patch pooling also averages out high-frequency noise before projection — a
  free smoothing regularizer.
- **Cost:** ~25 lines; use `nn.AvgPool1d` (or an unfold + `Linear`) plus one output linear.
- **Risk:** Low. `P` is a hyperparameter (e.g. `P=8` for ETT, `P=16` for Traffic).

### J. Hierarchical multi-scale decomposition
- **Idea:** Replace the single `series_decomp` with a **pyramid**: extract trends at 2–3 kernel
  sizes (e.g. `k ∈ {5, 25, 101}`) to build a hierarchy `t₁ ⊂ t₂ ⊂ t₃`. Compute residuals
  `r₀ = x − t₁`, `r₁ = t₁ − t₂`, `r₂ = t₂ − t₃`, feed each stream through its own **small**
  linear predictor (share weights across scales for maximum regularization), and sum the outputs.
- **Why it helps:** Different scales carry different periodicities (intraday, daily, weekly).
  Each per-scale predictor sees a simpler, narrower-band signal and therefore needs fewer
  parameters than the single wideband linear layer it replaces. This is an architecture change
  to the *front end* — decomposition capacity replaces predictor capacity.
- **Cost:** ~40 lines: extend `series_decomp` to output a list; add K matching predictors.
- **Risk:** Low if per-scale predictors are kept small and (optionally) share weights.

---

## 3. Recommended Experimentation Order

1. **A (RevIN)** — done; baseline is now RDLinear.
2. **F (low-rank predictor)** — start here for RDLinear. Cheapest architecture change,
   directly shrinks the dominant weight matrix, and every other direction can be layered on top.
3. **H (Fourier basis output)** — strongest inductive bias, further parameter reduction.
4. **G (RFFT predictor)** — a more expressive frequency-domain alternative to H; try when H
   under-fits.
5. **I (patch projection)** — good on high-frequency, high-channel data (Weather, ETTm*).
6. **J (multi-scale decomposition)** — layer on top of F / H once a single-scale winner is
   picked.

Each step should be measured against the **RDLinear baseline at the same look-back** so gains
are attributable to the architecture change alone.

---

## 4. Evaluation Protocol

- **Datasets:** ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Exchange-Rate, ILI.
- **Horizons:** `pred_len ∈ {96, 192, 336, 720}` (ILI: `{24, 36, 48, 60}`).
- **Metrics:** MSE and MAE on the test split (the repo's standard reporting).
- **Baseline:** Vanilla DLinear at a fixed look-back (recommend `seq_len = 336`), same seed
  (`fix_seed = 2021`, [run_longExp.py:8](run_longExp.py#L8)) and optimizer settings.
- **Controls:** Change one factor at a time; keep `--features M`, batch size, epochs, and early
  stopping fixed across comparisons.
- **Harness:** Reuse [scripts/EXP-LongForecasting/Linear/](scripts/EXP-LongForecasting/Linear/),
  swapping `--model` (and `--seq_len` as the experiment requires).
- **Report:** A table of MSE/MAE per dataset × horizon vs. the baseline, plus the % improvement,
  mirroring Table 2 of the paper.
