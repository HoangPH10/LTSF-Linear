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


class Model(nn.Module):
    """
    Reversible Instance Normalization + Decomposition-Linear (RDLinear).

    Wraps the DLinear decomposition model with reversible instance normalization
    (RevIN): each input instance is normalized per-channel using statistics from
    the look-back window, the forecast is produced in normalized space, and the
    output is de-normalized using the same statistics.

    Two normalization modes (``configs.revin_mode``):
      - 'std'  : subtract the look-back mean and divide by std (full RevIN).
      - 'mean' : subtract the look-back mean only (NLinear-style, no scaling).
    An optional learnable affine transform (``configs.revin_affine``) is applied
    in the normalized space, as in the original RevIN paper.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size (respects --moving_avg if provided)
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(kernel_size)
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
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

        x = self._denormalize(x, means, stdev)
        return x
