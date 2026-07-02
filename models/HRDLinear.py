import math

import torch
import torch.nn as nn

from models.RDLinear import series_decomp


class Model(nn.Module):
    """
    Fourier + polynomial basis-output RDLinear (HRDLinear) -- Plan N1.

    Inherits RDLinear's front-end (RevIN + series_decomp) but replaces both DMS
    heads with fixed-basis projections. The trainable Linears only produce
    coefficients; the horizon expansion is a matmul against a precomputed basis
    stored as a buffer.

    Seasonal head : Linear(seq_len -> 2B) coefficients c_s
                    forecast = c_s @ Phi_sin.T,  Phi_sin: [pred_len, 2B]
                    with columns [cos(2 pi b t), sin(2 pi b t)] for b = 1..B
                    and t = arange(pred_len) / pred_len.

    Trend head    : Linear(seq_len -> D+1) coefficients c_t
                    forecast = c_t @ Phi_poly.T, Phi_poly: [pred_len, D+1]
                    with columns [1, t, t^2, ..., t^D].

    Params per branch drop from seq_len * pred_len to seq_len * (2B) and
    seq_len * (D+1) respectively; the hypothesis space is hard-constrained to a
    low-order sinusoid + low-order polynomial, which rules out high-frequency
    noise memorisation.

    Extra config:
      - configs.fourier_b : number of Fourier harmonics for the seasonal head
                            (default 6, capped at pred_len // 2).
      - configs.poly_deg  : polynomial degree for the trend head (default 2).
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        # RevIN configuration (mirrors RDLinear)
        self.revin_mode = getattr(configs, 'revin_mode', 'std')
        self.revin_affine = getattr(configs, 'revin_affine', False)
        self.revin_eps = getattr(configs, 'revin_eps', 1e-5)
        if self.revin_affine:
            self.affine_weight = nn.Parameter(torch.ones(self.channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.channels))

        B = int(getattr(configs, 'fourier_b', 6))
        D = int(getattr(configs, 'poly_deg', 2))
        B = max(1, min(B, self.pred_len // 2))
        D = max(0, D)
        self.B = B
        self.D = D
        self.n_fourier = 2 * B
        self.n_poly = D + 1

        t = torch.arange(self.pred_len, dtype=torch.float32) / float(self.pred_len)

        fourier_cols = []
        for b in range(1, B + 1):
            fourier_cols.append(torch.cos(2.0 * math.pi * b * t))
            fourier_cols.append(torch.sin(2.0 * math.pi * b * t))
        self.register_buffer('fourier_basis', torch.stack(fourier_cols, dim=1))  # [pred_len, 2B]

        poly_cols = [t.pow(d) for d in range(D + 1)]
        self.register_buffer('poly_basis', torch.stack(poly_cols, dim=1))        # [pred_len, D+1]

        if self.individual:
            self.Coef_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.n_fourier) for _ in range(self.channels)]
            )
            self.Coef_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.n_poly) for _ in range(self.channels)]
            )
        else:
            self.Coef_Seasonal = nn.Linear(self.seq_len, self.n_fourier)
            self.Coef_Trend = nn.Linear(self.seq_len, self.n_poly)

    def _normalize(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        if self.revin_mode == 'std':
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.revin_eps).detach()
            x = x / stdev
        else:
            stdev = None
        if self.revin_affine:
            x = x * self.affine_weight + self.affine_bias
        return x, means, stdev

    def _denormalize(self, x, means, stdev):
        if self.revin_affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.revin_eps)
        if self.revin_mode == 'std':
            x = x * stdev
        x = x + means
        return x

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x, means, stdev = self._normalize(x)

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            cs = torch.zeros(
                [seasonal_init.size(0), self.channels, self.n_fourier],
                dtype=seasonal_init.dtype, device=seasonal_init.device,
            )
            ct = torch.zeros(
                [trend_init.size(0), self.channels, self.n_poly],
                dtype=trend_init.dtype, device=trend_init.device,
            )
            for i in range(self.channels):
                cs[:, i, :] = self.Coef_Seasonal[i](seasonal_init[:, i, :])
                ct[:, i, :] = self.Coef_Trend[i](trend_init[:, i, :])
        else:
            cs = self.Coef_Seasonal(seasonal_init)  # [B, C, 2B]
            ct = self.Coef_Trend(trend_init)        # [B, C, D+1]

        seasonal_output = torch.matmul(cs, self.fourier_basis.T)  # [B, C, pred_len]
        trend_output = torch.matmul(ct, self.poly_basis.T)        # [B, C, pred_len]

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # [B, pred_len, C]

        x = self._denormalize(x, means, stdev)
        return x
