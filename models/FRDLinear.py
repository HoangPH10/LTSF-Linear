import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class FreqLinear(nn.Module):
    """
    Frequency-domain predictor for a single (per-channel or shared) branch.

    Pipeline:
        x (seq_len) --rfft--> X (seq_len//2 + 1 complex bins)
                     take lowest K bins
                     complex linear map W in C^{K_out x K_in}
                     zero-pad up to pred_len//2 + 1
                     --irfft--> y (pred_len)

    The complex weight is stored as two real matrices (real / imaginary parts)
    so the module trains with standard real-valued autograd.
    """
    def __init__(self, seq_len, pred_len, K):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        in_bins = seq_len // 2 + 1
        out_bins = pred_len // 2 + 1
        K = min(int(K), in_bins, out_bins)
        self.K = max(K, 1)
        self.out_bins = out_bins

        scale = 1.0 / self.K
        self.weight_real = nn.Parameter(torch.randn(self.K, self.K) * scale)
        self.weight_imag = nn.Parameter(torch.randn(self.K, self.K) * scale)

    def forward(self, x):
        # x: [..., seq_len]
        X = torch.fft.rfft(x, dim=-1)              # [..., in_bins]
        X_low = X[..., :self.K]                    # [..., K]

        weight = torch.complex(self.weight_real, self.weight_imag)  # [K, K]
        Y_low = torch.matmul(X_low, weight.T)      # [..., K]

        pad = self.out_bins - self.K
        if pad > 0:
            Y_full = F.pad(Y_low, (0, pad))
        else:
            Y_full = Y_low
        y = torch.fft.irfft(Y_full, n=self.pred_len, dim=-1)  # [..., pred_len]
        return y


class Model(nn.Module):
    """
    Frequency-domain RDLinear (FRDLinear) -- Plan G.

    Inherits RDLinear's front-end (RevIN + series_decomp) and trend predictor,
    but replaces the seasonal linear layer with a low-pass complex-linear map
    in the frequency domain. Learnable params for the seasonal branch drop from
    seq_len * pred_len to K * K complex weights (2 * K * K real values).

    Extra config:
      - configs.freq_k : number of low-frequency bins to keep. Default 32.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        # RevIN configuration (same defaults as RDLinear)
        self.revin_mode = getattr(configs, 'revin_mode', 'std')
        self.revin_affine = getattr(configs, 'revin_affine', False)
        self.revin_eps = getattr(configs, 'revin_eps', 1e-5)
        if self.revin_affine:
            self.affine_weight = nn.Parameter(torch.ones(self.channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.channels))

        freq_k = getattr(configs, 'freq_k', 32)

        if self.individual:
            self.Freq_Seasonal = nn.ModuleList(
                [FreqLinear(self.seq_len, self.pred_len, freq_k) for _ in range(self.channels)]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            self.Freq_Seasonal = FreqLinear(self.seq_len, self.pred_len, freq_k)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

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
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype, device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype, device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Freq_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Freq_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # [B, pred_len, C]

        x = self._denormalize(x, means, stdev)
        return x
