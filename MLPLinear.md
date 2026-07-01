# MLPLinear vs. RDLinear

`MLPLinear` extends `RDLinear` with plan **C** from
[Improvement_plan.md](Improvement_plan.md) — *mild non-linearity / added capacity*.
It keeps every design decision from RDLinear (RevIN normalization, series
decomposition, optional per-channel `individual` mode, optional learnable
affine) and only changes the **per-branch predictor**: each single
`Linear(seq_len → pred_len)` is replaced by a 2-layer MLP that runs in
parallel with a linear skip, so the model degrades gracefully to RDLinear.

Files: [models/RDLinear.py](models/RDLinear.py) · [models/MLPLinear.py](models/MLPLinear.py)

---

## 1. Architecture side-by-side

Both models share the same outer pipeline:

```
x  →  RevIN normalize  →  series_decomp  →  (seasonal / trend predictor)
                                            +  sum  →  RevIN denormalize  →  ŷ
```

The only difference is the block marked *predictor*:

| Stage                       | RDLinear                                        | MLPLinear                                                                  |
| --------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------- |
| RevIN normalization         | `mean` / `std` + optional affine (`γ, β`)       | **identical**                                                              |
| Decomposition               | `series_decomp(kernel_size=configs.moving_avg)` | **identical**                                                              |
| Seasonal predictor          | `nn.Linear(seq_len, pred_len)`                  | `BranchMLP` **or** `nn.Linear`, per `--mlp_branches`                       |
| Trend predictor             | `nn.Linear(seq_len, pred_len)`                  | `BranchMLP` **or** `nn.Linear`, per `--mlp_branches`                       |
| `--individual` mode         | `ModuleList` of `Linear` per channel            | `ModuleList` of the chosen predictor per channel                           |
| RevIN denormalization       | inverse of `normalize`                          | **identical**                                                              |

`BranchMLP` is:

```
      ┌─────────────────────────────────────────────┐
x ──► │  Linear(seq_len → hidden) → GELU → Dropout  │──► fc2 ─┐
      │  Linear(hidden → pred_len)  (fc2, zero-init)│         │
      └─────────────────────────────────────────────┘         +──► ŷ_branch
                                                              │
      Linear(seq_len → pred_len)  (skip)  ────────────────────┘
```

Key init detail: `fc2.weight` and `fc2.bias` are **zero-initialized**, so the
MLP branch contributes exactly `0` at step 0 and the whole model starts as
RDLinear. Capacity is unlocked only as gradients push `fc2` away from zero,
which sharply reduces the overfitting risk called out in plan C.

---

## 2. Parameter & compute delta

Let `L = seq_len`, `H = hidden`, `P = pred_len`, `C = channels`.

| Component                  | RDLinear params / branch      | MLPLinear params / branch (MLP)                    |
| -------------------------- | ----------------------------- | -------------------------------------------------- |
| Linear predictor           | `L·P + P`                     | — (replaced)                                       |
| Skip                       | —                             | `L·P + P`                                          |
| `fc1` (`L → H`)            | —                             | `L·H + H`                                          |
| `fc2` (`H → P`)            | —                             | `H·P + P`                                          |
| **Per branch (MLP)**       | `L·P + P`                     | `L·P + L·H + H·P + 2P + H`                         |

With `L = 336`, `P = 96`, `H = 512`: `RDLinear ≈ 32K` vs.
`MLPLinear (MLP branch) ≈ 253K` per branch. In `--individual` mode this
multiplies by `C`, so on Traffic (`C = 862`) `mlp_branches=both` is *very*
heavy — the default `mlp_branches=seasonal` keeps the trend branch linear
and cuts the extra cost roughly in half. For channel-rich datasets prefer
smaller `--mlp_hidden` (e.g. 128–256) with `--individual` off, or run the
model in shared-weight mode.

---

## 3. Configuration differences

MLPLinear inherits every `RDLinear` config (`--revin_mode`, `--revin_affine`,
`--revin_eps`, `--moving_avg`, `--individual`) and adds three CLI flags in
[run_longExp.py](run_longExp.py):

| Flag              | Type    | Default      | Meaning                                                                |
| ----------------- | ------- | ------------ | ---------------------------------------------------------------------- |
| `--mlp_hidden`    | `int`   | `512`        | Hidden width of the per-branch MLP.                                    |
| `--mlp_dropout`   | `float` | `0.1`        | Dropout applied between GELU and the second linear.                    |
| `--mlp_branches`  | `str`   | `seasonal`   | Which branch uses the MLP: `both` \| `seasonal` \| `trend`.            |

Everything else — training loop, data pipeline, metrics — is unchanged.
The `'Linear' in args.model` routing in [exp/exp_main.py](exp/exp_main.py#L70)
picks up `MLPLinear` automatically because the name contains `"Linear"`.

---

## 4. When to prefer which

| Situation                                                                     | Use            |
| ----------------------------------------------------------------------------- | -------------- |
| First run on any dataset / need a stable, low-variance baseline               | **RDLinear**   |
| Series shows regime shifts, change points, or non-linear seasonality (ETT, Weather, Exchange at long horizon) | **MLPLinear** with `--mlp_branches both` |
| Channel-rich, strongly periodic (Electricity, Traffic)                         | Try MLPLinear with small `--mlp_hidden` (128–256); expect marginal gains |
| Small-sample / short series (ILI)                                              | Prefer RDLinear; if using MLPLinear, drop `--mlp_hidden` to 64–128 and raise `--mlp_dropout` to 0.2–0.3 |

---

## 5. Backwards compatibility

Setting `--mlp_hidden 0` is not supported, but the design ensures MLPLinear
**starts numerically identical to RDLinear** thanks to the zero-init of
`fc2`. If you want to reproduce RDLinear exactly, either use the
`RDLinear` model directly or set `--mlp_branches` to a branch and freeze
`fc2` — the simpler path is just to run `--model RDLinear`.
