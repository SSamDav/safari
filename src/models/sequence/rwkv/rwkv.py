import torch
import torch.nn as nn
import os
import math 
import logging

from types import SimpleNamespace
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class RWKV_TimeMix_RWKV5_Preview(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.head_size = config.head_size
        self.n_head = config.n_embd // self.head_size

        assert config.n_embd % self.n_head == 0
        self.head_size_divisor = math.sqrt(self.head_size)

        self.chunk_len = 512
        assert config.ctx_len % self.chunk_len == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -8 + 7 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            if 'r2' in os.environ["RWKV_MY_TESTING"]:
                self.time_faaaa = nn.Parameter(torch.ones(self.n_head) * 0.05)
            else:
                self.time_first = nn.Parameter(torch.ones(self.n_head) * (-3.0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd)

    @torch.jit.script_method
    def jit_func(self, x):
        B, TT, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
        k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
        v = self.value(xv).view(B, TT, self.n_head, self.head_size).transpose(1, 2)                 # BTC -> BHTS

        return r, k, v

    @torch.jit.script_method
    def jit_func_2(self, r, k, v, w, wk, wb, ws):
        B, H, TT, S = r.size()
        T = self.chunk_len

        s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # state
        x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output

################################################################################
########
        for i in range(TT // T):
            rr = r[:, :, i*T:i*T+T, :]
            kk = k[:, :, :, i*T:i*T+T]
            vv = v[:, :, i*T:i*T+T, :]

            x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb

            s = ws * s + (kk * wk) @ vv
########
################################################################################
        
        x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC
        x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S)
        return self.output(x)
    
    def forward(self, x):
        H = self.n_head
        T = self.chunk_len

        r, k, v = self.jit_func(x)

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)
        u = self.time_faaaa.float().unsqueeze(-1)

################################################################################
########
        ws = w.pow(T).reshape(1, H, 1, 1)

        ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)

        wk = w.reshape(1, H, 1, T)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(1, H, T, T)
########
################################################################################

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)
        return self.jit_func_2(r, k, v, w, wk, wb, ws)
    

class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
            if config.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,config.my_pos_emb,config.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((config.my_pos_emb,1,config.n_embd)))

        if self.layer_id == 0 and self.config.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(config, 0)
        else:
            self.att = RWKV_TimeMix_RWKV5_Preview(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)

        if config.dropout > 0:
            self.drop0 = nn.Dropout(p = config.dropout)
            self.drop1 = nn.Dropout(p = config.dropout)
        
    def forward(self, x, x_emb=None):
        config = self.config
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if config.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if self.config.dropout == 0:
            if self.layer_id == 0 and config.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and config.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        return x


class RWKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        if type(config) is dict:
            config = SimpleNamespace(**config)
        self.config = config

        # self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.d_model = self.config.n_embd
        self.blocks = nn.ModuleList([Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        #self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.head = nn.Linear(config.n_embd, config.n_embd , bias=False)
            
        if config.dropout > 0:
            self.drop0 = nn.Dropout(p = config.dropout)

        # self.ctx_len = config.ctx_len

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, state=None, **kwargs):
        self.step += 1

        if self.config.dropout > 0:
            x = self.drop0(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)

        return x, state

    @property
    def d_state(self):
        return self.n_layers * self.d_model

    @property
    def d_output(self):
        return self.d_model