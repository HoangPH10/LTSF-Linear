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
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    """
    Multi-scale series decomposition: trend is a softmax-weighted mixture of
    moving averages with several kernel sizes (FEDformer-style). The mixture
    weights are produced per sample/time-step/channel by a tiny linear gate
    from the raw input value, so different regimes can prefer different scales.
    """
    def __init__(self, kernel_sizes):
        super(series_decomp_multi, self).__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = list(kernel_sizes)
        self.moving_avgs = nn.ModuleList(
            [moving_avg(k, stride=1) for k in self.kernel_sizes]
        )
        self.gate = nn.Linear(1, len(self.kernel_sizes))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        trends = [ma(x).unsqueeze(-1) for ma in self.moving_avgs]
        trends = torch.cat(trends, dim=-1)  # [B, L, C, K]
        weights = F.softmax(self.gate(x.unsqueeze(-1)), dim=-1)  # [B, L, C, K]
        moving_mean = torch.sum(trends * weights, dim=-1)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Multi-scale Decomposition-Linear (MDLinear) with optional Reversible
    Instance Normalization (RevIN).

    Same head as DLinear (per-channel linear maps for seasonal + trend), but
    the trend is extracted with a learnable mixture over several moving-average
    kernel sizes (``configs.decomp_kernels``) instead of a single fixed kernel.

    When enabled, each input instance is normalized per-channel using look-back
    statistics, the forecast is produced in normalized space, then de-normalized
    with the same statistics. Modes (``configs.revin_mode``):
      - 'std'  : subtract mean and divide by std (full RevIN).
      - 'mean' : subtract mean only (NLinear-style).
    An optional learnable affine (``configs.revin_affine``) is applied in
    normalized space, as in the original RevIN paper.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        kernel_sizes = getattr(configs, 'decomp_kernels', None)
        if not kernel_sizes:
            kernel_sizes = [getattr(configs, 'moving_avg', 25)]
        # Enforce odd kernels for symmetric padding behavior
        self.kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
        self.decompsition = series_decomp_multi(kernel_sizes)
        self.individual = configs.individual
        self.channels = configs.enc_in

        # RevIN configuration (kept backward-compatible via getattr)
        self.revin_mode = getattr(configs, 'revin_mode', 'std')
        self.revin_affine = getattr(configs, 'revin_affine', False)
        self.revin_eps = getattr(configs, 'revin_eps', 1e-5)
        if self.revin_affine:
            self.affine_weight = nn.Parameter(torch.ones(self.channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.channels))

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def _normalize(self, x):
        # x: [Batch, Input length, Channel]
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
        # x: [Batch, Output length, Channel]
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
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # [Batch, Output length, Channel]
        
        x = self._denormalize(x, means, stdev)
        return x
