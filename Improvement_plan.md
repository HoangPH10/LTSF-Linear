# Improvement Plan for RDLinear (Architecture-side Overfitting Reduction)

A plan of **architecture changes** that make RDLinear structurally harder to overfit.
Motivated by the failure mode of prior variants (MLPLinear / CDLinear added capacity;
LRDLinear / FRDLinear replaced components but still over-fit test data): the underlying
problem is the two DMS heads with `seq_len · pred_len` parameters each (up to `2 · 336 · 720
≈ 484K` per channel, times `C` when `--individual` is on) — far more capacity than the ETT,
Weather, Exchange or ILI training sets can support.

Every idea below **replaces** a component of RDLinear ([models/RDLinear.py](models/RDLinear.py))
using one (or more) of three overfitting-reducing levers:

1. **Parameter cut** — fewer trainable weights.
2. **Weight sharing** — one small predictor reused across channels / horizon positions / scales.
3. **Hypothesis-space constraint** — lock the output (or input) to a low-order fixed basis so
   the model cannot express high-frequency noise.

No branch is *added* to RDLinear; RevIN and `series_decomp` stay in place unless a proposal
explicitly restructures the front end (N4).


---

## 1. Background & Current Bottleneck

RDLinear does direct multi-step (DMS) forecasting with two `Linear(seq_len → pred_len)` heads
(seasonal + trend), optionally per channel via `--individual`. Even in the shared-weights
setting the parameter count is `2 · seq_len · pred_len` ≈ 484K at `seq_len=336, pred_len=720`,
and with `--individual` it multiplies by `C` (up to 862 on Traffic). This is orders of
magnitude larger than a typical training set, so the DMS head trivially memorises
batch-level noise.

What we observe in the training logs:

- Train loss keeps decreasing across all 10 epochs.
- Validation loss bottoms out at epoch 2–4, then flattens or rises.
- The best-val checkpoint is preserved by early stopping, but the val/test **gap** relative
  to the train loss keeps widening — i.e. the model still overfits *within* the surviving
  epochs.

So the highest-leverage moves are **training-side regularisers**, not more model capacity.

Two implementation facts make experimentation cheap:

- The forward path is selected by `'Linear' in args.model` in
  [exp/exp_main.py](exp/exp_main.py#L70); any model whose name contains "Linear" is driven with
  the single-input `self.model(batch_x)` call.
- Optimizer, criterion, and early-stopping live in
  [exp/exp_main.py](exp/exp_main.py) (`_select_optimizer`, `_select_criterion`, and the
  `EarlyStopping` handler in the training loop) — all the changes below plug in there.


---

## 2. Architecture Interventions

Baseline reference throughout: shared-weight RDLinear at `seq_len=336, pred_len=720` has
`2 · 336 · 720 = 483,840` head parameters. Each entry states the resulting parameter count so
the capacity cut is explicit. Each: *idea → why reduces overfitting → params → cost → risk*.

### N1. Fourier + polynomial basis output head (HRDLinear) — *highest priority*
- **Idea:** Seasonal head outputs `2B` sine / cosine coefficients (`B ∈ {4, 6, 8}`); trend
  head outputs `D+1` polynomial coefficients (`D ∈ {2, 3}`). Forecast is
  `ŷ = Φ_sin · c_s + Φ_poly · c_t`, with both `Φ` matrices fixed and stored via
  `register_buffer`. N-BEATS' interpretable blocks, made linear.
- **Why:** *Both* param cut *and* hypothesis constraint. Params drop to
  `seq_len · (2B + D + 1)` ≈ `336 · 15 = 5,040` at `B=6, D=2` — **~96× smaller**. The output
  space is hard-constrained to a low-order sinusoid + low-order polynomial, so high-frequency
  noise is architecturally unreachable. Ideal fit for the seasonal + trend decomposition.
- **Cost:** ~30 lines: two Linears + two fixed basis buffers.
- **CLI:** `--fourier_b`, `--poly_deg`.
- **Risk:** Low. Under-fitting is possible on fast-changing series; the polynomial trend
  branch absorbs slow non-periodic components. Falls back to plain Linear if `B` is set to
  `pred_len // 2` and `D = pred_len - 1`.

### N2. Chunked / tiled DMS with shared per-chunk weights (CRDLinear)
- **Idea:** Split `pred_len` into `pred_len / H` chunks of length `H` (e.g. `H=24`). Use one
  shared `Linear(seq_len → H)` for every chunk and add a small learnable positional bias
  `[pred_len / H, H]` that captures the horizon-dependent offset per chunk. Applied to both
  seasonal and trend branches (each with its own shared Linear + bias).
- **Why:** Weight sharing across horizon positions. Per branch:
  `seq_len · H + pred_len = 336 · 24 + 720 = 8,784` params — **~55× smaller** than the RDLinear
  head. The prior "nearby horizon steps are predicted by the same function of the look-back"
  is exactly the invariance we want on periodic data.
- **Cost:** ~20 lines.
- **CLI:** `--chunk_h`.
- **Risk:** Low; requires `pred_len % H == 0`. Sweep `H ∈ {12, 24, 48}`.

### N3. Patch pooling on the input (PRDLinear)
- **Idea:** After RevIN and decomposition, average-pool each branch's input into non-
  overlapping patches of length `P` (`P ∈ {8, 16, 24}`), giving `seq_len / P` tokens. Feed
  through `Linear(seq_len/P → pred_len/P)`, then upsample by repeat / one shared
  `Linear(1 → P)`.
- **Why:** Both param cut (per branch: `(seq_len / P) · (pred_len / P) ≈ 336/16 · 720/16 ≈ 945`,
  **~500× smaller**) and implicit low-pass filtering — the average pool discards the high-
  frequency components that a linear head would otherwise memorise. PatchTST's tokenisation
  applied to a linear model.
- **Cost:** ~25 lines with `nn.AvgPool1d` + one head Linear + `F.interpolate`.
- **CLI:** `--patch_len`.
- **Risk:** Low. `P` too large under-fits fine detail (ETTm*); `P` too small recovers RDLinear.

### N4. Multi-scale decomposition with tied per-scale predictor (MRDLinear)
- **Idea:** Replace the single `series_decomp` with a pyramid of moving averages
  `k ∈ {5, 25, 101}`, giving trends `t₁ ⊂ t₂ ⊂ t₃` and narrow-band residuals
  `r₀ = x − t₁, r₁ = t₁ − t₂, r₂ = t₂ − t₃`. All four streams share **one** small
  `Linear(seq_len → pred_len)` (bottlenecked, e.g. via a rank-`r` factorisation or reduced
  `pred_len` output followed by upsample). Sum the outputs.
- **Why:** Weight sharing across scales. Instead of two independent heads, one predictor is
  reused four times — an implicit consistency prior. Each narrower-band signal is easier to
  fit than the wideband seasonal, so a smaller shared head suffices. Best paired with N1's
  basis constraint on the shared head for maximum regularisation.
- **Cost:** ~40 lines: extend `series_decomp` to output a list; one tied head.
- **CLI:** `--ms_kernels "5,25,101"`.
- **Risk:** Low if the tied predictor is kept small (combine with N1 or N2). Bare use with a
  full `Linear(seq_len → pred_len)` recovers the baseline capacity and won't help.

### N5. Fixed DCT input basis (DRDLinear)
- **Idea:** After RevIN, apply a fixed (non-trainable) DCT matrix to the look-back, keep the
  first `K` coefficients (`K ∈ {16, 32, 64}`), and predict via `Linear(K → pred_len)` per
  branch. Store the DCT matrix via `register_buffer`.
- **Why:** Reduces the *effective* input dimension from `seq_len` to `K`, so head params
  become `K · pred_len` ≈ `32 · 720 = 23,040` per branch — **~10× smaller**. The DCT compresses
  smooth signals into a few low-frequency coefficients; the head simply cannot see the
  high-frequency components that noise lives in. Dual of N1 (basis on input vs. output).
- **Cost:** ~20 lines: precompute the DCT matrix; one matmul; one Linear.
- **CLI:** `--dct_k`.
- **Risk:** Low–moderate. Aperiodic series (Exchange, ILI) can lose useful high-frequency
  detail if `K` is set too aggressively.

---

## 3. Recommended Experimentation Order

1. **N1 (Fourier + polynomial basis)** — largest parameter cut *and* strongest inductive
   bias. Start here.
2. **N2 (chunked DMS)** — orthogonal to N1; weight sharing across horizon. Easy to combine.
3. **N3 (patch pooling)** — additional ×P cut on the input side; especially useful for
   high-frequency datasets (Weather, ETTm*).
4. **N5 (fixed DCT input)** — a smoother, denoised input version of N3; try if N3 underfits.
5. **N4 (multi-scale + tied head)** — layer on top of the winner of N1–N3; tie the small head
   from that winner across the scales produced by the pyramid decomposition.

Rules: change one component at a time; keep every other flag identical to the current
RDLinear scripts; compare against **shared-weight** RDLinear (drop `--individual`) at the
same `seq_len` — that is the fair baseline once you're committing to a capacity cut.
Combining N1 + N2 is expected to give the best single configuration; run it explicitly.

---

## 4. Evaluation Protocol

- **Datasets:** ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Exchange-Rate, ILI.
- **Horizons:** `pred_len ∈ {96, 192, 336, 720}` (ILI: `{24, 36, 48, 60}`).
- **Metrics:** MSE and MAE on the test split (the repo's standard reporting).
- **Baseline:** Current **RDLinear** at `seq_len = 336` (ILI: 104), same seed
  (`fix_seed = 2021`, [run_longExp.py:8](run_longExp.py#L8)) and per-dataset settings from
  [scripts/EXP-LongForecasting/RDLinear/](scripts/EXP-LongForecasting/RDLinear/).
- **Controls:** Change one factor at a time; keep `--features M`, batch size, epochs, and
  early stopping fixed across comparisons unless the intervention itself changes them.
- **Harness:** Reuse [scripts/EXP-LongForecasting/RDLinear/](scripts/EXP-LongForecasting/RDLinear/),
  toggling the CLI arg for the intervention under test.
- **Overfitting metric:** In addition to test MSE/MAE, log the *gap* `val_loss − train_loss`
  at the best-val epoch. Interventions should shrink this gap; a shrinking gap without a
  test-MSE improvement means the intervention is under-regularising or over-regularising.
- **Report:** A table of MSE/MAE per dataset × horizon vs. the RDLinear baseline, plus the %
  improvement, mirroring Table 2 of the paper.
