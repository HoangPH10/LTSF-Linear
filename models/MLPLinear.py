import torch
import torch.nn as nn


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


class BranchMLP(nn.Module):
    """
    2-layer MLP (Linear -> GELU -> Dropout -> Linear) with a parallel linear
    skip connection from seq_len -> pred_len. The skip preserves DLinear's
    behavior at initialization and stabilizes training; the MLP branch adds
    capacity to model non-linear / change-point dynamics.
    """
    def __init__(self, seq_len, pred_len, hidden, dropout):
        super(BranchMLP, self).__init__()
        self.fc1 = nn.Linear(seq_len, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, pred_len)
        self.skip = nn.Linear(seq_len, pred_len)
        # Zero-init the MLP output so the model starts as a pure linear map.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.skip(x) + self.fc2(self.drop(self.act(self.fc1(x))))


class Model(nn.Module):
    """
    MLPLinear: RevIN + series-decomposition + per-branch 2-layer MLP with
    linear skip. Implements plan C (mild non-linearity / added capacity)
    from the improvement plan, layered on RDLinear's RevIN backbone.

    Relevant configs (all optional, backward-compatible via getattr):
      - moving_avg (int)        : decomposition kernel size (default 25)
      - mlp_hidden (int)        : MLP hidden width (default 512)
      - mlp_dropout (float)     : dropout inside the MLP (default 0.1)
      - mlp_branches (str)      : 'both' | 'seasonal' | 'trend'
                                  which branch(es) get MLP capacity;
                                  the other stays a plain Linear
                                  (default 'seasonal' — trend is smooth
                                  and rarely benefits from non-linearity)
      - revin_mode (str)        : 'std' | 'mean'  (default 'std')
      - revin_affine (bool)     : learnable affine in normalized space
      - revin_eps (float)       : numerical stability epsilon
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        hidden = getattr(configs, 'mlp_hidden', 512)
        dropout = getattr(configs, 'mlp_dropout', 0.1)
        branches = getattr(configs, 'mlp_branches', 'seasonal')
        self.mlp_seasonal = branches in ('both', 'seasonal')
        self.mlp_trend = branches in ('both', 'trend')

        # RevIN configuration (kept backward-compatible via getattr)
        self.revin_mode = getattr(configs, 'revin_mode', 'std')
        self.revin_affine = getattr(configs, 'revin_affine', False)
        self.revin_eps = getattr(configs, 'revin_eps', 1e-5)
        if self.revin_affine:
            self.affine_weight = nn.Parameter(torch.ones(self.channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.channels))

        def make_seasonal():
            return BranchMLP(self.seq_len, self.pred_len, hidden, dropout) \
                if self.mlp_seasonal else nn.Linear(self.seq_len, self.pred_len)

        def make_trend():
            return BranchMLP(self.seq_len, self.pred_len, hidden, dropout) \
                if self.mlp_trend else nn.Linear(self.seq_len, self.pred_len)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList([make_seasonal() for _ in range(self.channels)])
            self.Linear_Trend = nn.ModuleList([make_trend() for _ in range(self.channels)])
        else:
            self.Linear_Seasonal = make_seasonal()
            self.Linear_Trend = make_trend()

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
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

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
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)

        x = self._denormalize(x, means, stdev)
        return x
