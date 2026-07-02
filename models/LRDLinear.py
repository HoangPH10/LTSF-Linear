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


class LowRankLinear(nn.Module):
    """
    Bottleneck factorization of ``Linear(in_features -> out_features)``:

        Linear(in_features -> rank, bias=False) -> [optional GELU]
            -> Linear(rank -> out_features, bias=True)

    Parameter count ``(in_features + out_features) * rank + out_features`` is
    much smaller than the dense ``in_features * out_features + out_features``
    whenever ``rank << min(in_features, out_features)``, giving a built-in
    regularizer while preserving the shape of the mapping.
    """
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class Model(nn.Module):
    """
    LRDLinear: RevIN + series-decomposition + per-branch **low-rank**
    (bottleneck) linear predictor. Implements plan F (low-rank / bottleneck
    linear predictor) from the improvement plan, on top of RDLinear's
    RevIN backbone.

    Relevant configs (all optional, backward-compatible via getattr):
      - moving_avg (int)      : decomposition kernel size (default 25)
      - lr_rank (int)         : bottleneck rank; if <=0 or >= min(seq_len,
                                pred_len) the factorization collapses to a
                                plain Linear layer (default 32)
      - lr_gelu (bool)        : insert a GELU between the two factors
                                (default False — plain factorization is
                                the safer starting point)
      - revin_mode (str)      : 'std' | 'mean' (default 'std')
      - revin_affine (bool)   : learnable affine in normalized space
      - revin_eps (float)     : numerical stability epsilon
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        rank = getattr(configs, 'lr_rank', 32)
        # If the requested rank is degenerate, fall back to a plain linear
        # so the model still matches RDLinear numerically.
        self.use_lowrank = rank > 0 and rank < min(self.seq_len, self.pred_len)

        self.revin_mode = getattr(configs, 'revin_mode', 'std')
        self.revin_affine = getattr(configs, 'revin_affine', False)
        self.revin_eps = getattr(configs, 'revin_eps', 1e-5)
        if self.revin_affine:
            self.affine_weight = nn.Parameter(torch.ones(self.channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.channels))

        def make_predictor():
            if self.use_lowrank:
                return LowRankLinear(self.seq_len, self.pred_len, rank)
            return nn.Linear(self.seq_len, self.pred_len)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList([make_predictor() for _ in range(self.channels)])
            self.Linear_Trend = nn.ModuleList([make_predictor() for _ in range(self.channels)])
        else:
            self.Linear_Seasonal = make_predictor()
            self.Linear_Trend = make_predictor()

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
