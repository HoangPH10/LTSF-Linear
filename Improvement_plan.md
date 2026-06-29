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

### B. Better / unfrozen decomposition
- **Idea (B1):** Respect the existing `--moving_avg` argument ([run_longExp.py:51](run_longExp.py#L51))
  instead of the hardcoded `kernel_size = 25` at [models/DLinear.py:48](models/DLinear.py#L48), and
  tune it per dataset.
- **Idea (B2):** Multi-scale "mixture of kernels" — extract trend with several moving-average
  kernels of different sizes and learn a weighted combination (the FEDformer trend mixture),
  capturing both slow and fast trends.
- **Why it helps:** Trend quality directly limits the trend-branch linear layer; a single fixed
  kernel is rarely optimal across datasets/horizons.
- **Cost:** B1 trivial; B2 low–moderate.
- **Implement:** Edit `series_decomp` / `moving_avg` in [models/DLinear.py](models/DLinear.py#L48);
  for B2 add a `series_decomp_multi` block.
- **Risk:** Low. Mostly a hyperparameter/inductive-bias change.

### C. Mild non-linearity / added capacity
- **Idea:** Replace each component's single linear layer with a small 2-layer MLP (with
  GELU/ReLU + dropout) or add a residual term, optionally only on the seasonal branch.
- **Why it helps:** Addresses the paper's stated weakness — a one-layer linear model cannot model
  change-point / regime dynamics. Extra capacity can fit non-linear seasonal structure.
- **Cost:** Moderate; more params and a real overfitting risk on small datasets (ETT, ILI, Exchange).
- **Implement:** New `models/MLPLinear.py`; expose hidden width + dropout as args; rely on early
  stopping ([exp/exp_main.py](exp/exp_main.py)).
- **Risk:** Moderate–high. Trades interpretability and may overfit; tune dropout and keep the MLP
  shallow.

### D. Cross-variate (channel-mixing) modeling
- **Idea:** Add a light layer that mixes information across channels — e.g. a low-rank linear map
  across the channel dimension, or a shared+individual hybrid — on top of the per-channel forecasts.
- **Why it helps:** DLinear forecasts each variate independently; high-dimensional, correlated
  datasets (Traffic = 862, Electricity = 321 channels) have exploitable inter-series structure.
- **Cost:** Moderate; parameter count grows with channel count.
- **Implement:** New variant (e.g. `models/CDLinear.py`); the `'Linear'`-in-name routing still
  applies. Consider low-rank factorization to bound parameters.
- **Risk:** High overfitting on channel-poor datasets (ETT has 7); best treated as dataset-specific.

### E. Loss & training tweaks — *orthogonal, stackable*
- **Idea:** Try MAE/Huber loss instead of MSE (`--loss`), tune the LR schedule
  (`--lradj`, `--learning_rate`), and/or weight the loss per horizon step (down-weight far steps).
- **Why it helps:** MSE is outlier-sensitive; robust losses can help noisy/aperiodic series
  (Exchange). Schedule tuning often yields a few % with no architectural change.
- **Cost:** Low. Mostly config; per-horizon weighting is a few lines in the training loop.
- **Implement:** `--loss`, `--lradj` args already exist ([run_longExp.py:71-72](run_longExp.py#L71));
  per-horizon weighting goes in [exp/exp_main.py](exp/exp_main.py).
- **Risk:** Low. Combine with any of the above.

---

## 3. Recommended Experimentation Order

1. **A (RevIN)** — best accuracy-per-effort; low risk.
2. **B (decomposition)** — cheap inductive-bias gains; stack on top of A.
3. **C / D** — higher-effort follow-ups; C for capacity-limited datasets, D for channel-rich ones.
4. **E (loss/training)** — stackable throughout; fold into every run.

Each step should be measured against the **vanilla DLinear baseline at the same look-back** so
gains are attributable.

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
  swapping `--model` (and `--seq_len` / `--moving_avg` / `--loss` as the experiment requires).
- **Report:** A table of MSE/MAE per dataset × horizon vs. the baseline, plus the % improvement,
  mirroring Table 2 of the paper.
