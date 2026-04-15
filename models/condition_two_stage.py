import math
import torch
from torch import nn


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed(length, dim):
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pos_embed = torch.zeros(length, dim, dtype=torch.float32)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)
    return pos_embed.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Attention dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"CrossAttention dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, query, context):
        bsz, q_len, dim = query.shape
        c_len = context.shape[1]
        q = self.q_proj(query).reshape(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(bsz, c_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(bsz, c_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, q_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GraphConvBlock(nn.Module):
    def __init__(self, num_nodes, hidden_size, dropout=0.0):
        super().__init__()
        self.adj = nn.Parameter(torch.eye(num_nodes))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(hidden_size, hidden_size * 4, dropout=dropout)

    def forward(self, x):
        adj = torch.softmax(self.adj, dim=-1)
        y = self.norm1(x)
        y = self.fc1(y)
        y = torch.einsum('ij,bjd->bid', adj, y)
        y = torch.nn.functional.gelu(y)
        y = self.fc2(y)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.out_proj = nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.out_proj(x)


class MotionTransformerTwoStage(nn.Module):
    def __init__(self,
                 input_feats,
                 cond_feats=None,
                 human_cond_joint_num=None,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 stage1_num_layers=None,
                 **kargs):
        super().__init__()
        del activation, kargs
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.input_feats = input_feats
        self.cond_feats = input_feats if cond_feats is None else cond_feats
        if self.input_feats % 3 != 0:
            raise ValueError(f"input_feats must be divisible by 3, got {self.input_feats}.")
        if self.cond_feats % 3 != 0:
            raise ValueError(f"cond_feats must be divisible by 3, got {self.cond_feats}.")
        self.input_joint_num = self.input_feats // 3
        self.cond_joint_num = self.cond_feats // 3
        self.human_cond_joint_num = self.input_joint_num if human_cond_joint_num is None else human_cond_joint_num
        self.robot_cond_joint_num = max(0, self.cond_joint_num - self.human_cond_joint_num)
        if self.human_cond_joint_num <= 0:
            raise ValueError(f"human_cond_joint_num must be > 0, got {self.human_cond_joint_num}.")
        if self.human_cond_joint_num > self.cond_joint_num:
            raise ValueError(
                f"human_cond_joint_num ({self.human_cond_joint_num}) exceeds cond joints ({self.cond_joint_num})."
            )

        if stage1_num_layers is None:
            self.stage1_num_layers = max(1, num_layers // 2)
        else:
            self.stage1_num_layers = int(stage1_num_layers)
        if not 1 <= self.stage1_num_layers < num_layers:
            raise ValueError(
                f"stage1_num_layers must be in [1, {num_layers - 1}], got {self.stage1_num_layers}."
            )

        self.x_embedder = nn.Linear(self.input_feats, latent_dim)
        self.t_embedder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.human_joint_embed = nn.Linear(3, latent_dim)
        self.robot_joint_embed = nn.Linear(3, latent_dim) if self.robot_cond_joint_num > 0 else None
        self.human_gcn = nn.ModuleList([
            GraphConvBlock(self.human_cond_joint_num, latent_dim, dropout=dropout)
            for _ in range(2)
        ])
        self.human_cond_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.cross_attn = CrossAttention(latent_dim, num_heads=num_heads, dropout=dropout) if self.robot_cond_joint_num > 0 else None
        self.cross_norm_q = nn.LayerNorm(latent_dim) if self.robot_cond_joint_num > 0 else None
        self.cross_norm_kv = nn.LayerNorm(latent_dim) if self.robot_cond_joint_num > 0 else None
        self.interaction_cond_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, latent_dim), requires_grad=False)
        self.cond_time_pos_embed = nn.Parameter(torch.zeros(1, num_frames, latent_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, num_heads, mlp_ratio=ff_size / latent_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_layer = FinalLayer(latent_dim, self.input_feats)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)

        pos_embed = get_1d_sincos_pos_embed(self.num_frames, self.latent_dim)
        self.pos_embed.data.copy_(pos_embed)
        self.cond_time_pos_embed.data.copy_(pos_embed)

        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.zeros_(self.t_embedder[0].bias)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)
        nn.init.zeros_(self.t_embedder[2].bias)

        for proj in (self.human_cond_proj, self.interaction_cond_proj):
            nn.init.normal_(proj[1].weight, std=0.02)
            nn.init.zeros_(proj[1].bias)
            nn.init.normal_(proj[3].weight, std=0.02)
            nn.init.zeros_(proj[3].bias)

        nn.init.xavier_uniform_(self.human_joint_embed.weight)
        nn.init.zeros_(self.human_joint_embed.bias)
        if self.robot_joint_embed is not None:
            nn.init.xavier_uniform_(self.robot_joint_embed.weight)
            nn.init.zeros_(self.robot_joint_embed.bias)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[1].weight)
            nn.init.zeros_(block.adaLN_modulation[1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_layer.out_proj.weight)
        nn.init.zeros_(self.final_layer.out_proj.bias)

    def encode_human_condition(self, mod):
        bsz, seq_len, _ = mod.shape
        human_dim = self.human_cond_joint_num * 3
        human_motion = mod[:, :, :human_dim].reshape(bsz, seq_len, self.human_cond_joint_num, 3)
        human_tokens = self.human_joint_embed(human_motion).reshape(
            bsz * seq_len, self.human_cond_joint_num, self.latent_dim
        )
        for gcn in self.human_gcn:
            human_tokens = gcn(human_tokens)
        human_tokens = human_tokens.reshape(bsz, seq_len, self.human_cond_joint_num, self.latent_dim)
        time_pe = self.cond_time_pos_embed[:, :seq_len].unsqueeze(2)
        human_tokens = human_tokens + time_pe
        human_time_tokens = human_tokens.mean(dim=2)
        human_global = self.human_cond_proj(human_time_tokens.mean(dim=1))
        return human_tokens, human_time_tokens, human_global

    def encode_interaction_condition(self, mod, human_tokens):
        if self.robot_cond_joint_num == 0:
            human_time_tokens = human_tokens.mean(dim=2)
            interaction_global = self.interaction_cond_proj(human_time_tokens.mean(dim=1))
            return human_time_tokens, interaction_global

        bsz, seq_len, _ = mod.shape
        human_dim = self.human_cond_joint_num * 3
        time_pe = self.cond_time_pos_embed[:, :seq_len].unsqueeze(2)
        robot_motion = mod[:, :, human_dim:].reshape(bsz, seq_len, self.robot_cond_joint_num, 3)
        robot_tokens = self.robot_joint_embed(robot_motion) + time_pe
        robot_tokens = robot_tokens.reshape(bsz, seq_len * self.robot_cond_joint_num, self.latent_dim)

        human_query = human_tokens.reshape(bsz, seq_len * self.human_cond_joint_num, self.latent_dim)
        human_query = human_query + self.cross_attn(
            self.cross_norm_q(human_query),
            self.cross_norm_kv(robot_tokens)
        )
        human_query = human_query.reshape(bsz, seq_len, self.human_cond_joint_num, self.latent_dim)
        interaction_time_tokens = human_query.mean(dim=2)
        interaction_global = self.interaction_cond_proj(interaction_time_tokens.mean(dim=1))
        return interaction_time_tokens, interaction_global

    def forward(self, x, timesteps, mod=None):
        bsz, seq_len, _ = x.shape
        if seq_len > self.num_frames:
            raise ValueError(f"Sequence length {seq_len} exceeds configured num_frames {self.num_frames}")

        x = self.x_embedder(x) + self.pos_embed[:, :seq_len]
        c_base = self.t_embedder(timestep_embedding(timesteps, self.latent_dim))

        stage1_time_tokens = None
        stage2_time_tokens = None
        c_stage1 = c_base
        c_stage2 = c_base

        if mod is not None:
            if mod.ndim != 3:
                raise ValueError(f"Condition mod must be 3D [B, T, D_cond], got shape {tuple(mod.shape)}.")
            if mod.shape[0] != bsz:
                raise ValueError(f"Condition batch mismatch: mod B={mod.shape[0]} vs x B={bsz}.")
            if mod.shape[1] != seq_len:
                raise ValueError(f"Condition time mismatch: mod T={mod.shape[1]} vs x T={seq_len}.")
            if mod.shape[2] != self.cond_feats:
                raise ValueError(
                    f"Condition feature mismatch: mod D={mod.shape[2]} vs expected cond_feats={self.cond_feats}."
                )
            human_tokens, stage1_time_tokens, human_global = self.encode_human_condition(mod)
            stage2_time_tokens, interaction_global = self.encode_interaction_condition(mod, human_tokens)
            c_stage1 = c_base + human_global
            c_stage2 = c_base + interaction_global
            x = x + stage1_time_tokens[:, :seq_len]

        for i, block in enumerate(self.blocks):
            if i == self.stage1_num_layers and stage2_time_tokens is not None:
                x = x + stage2_time_tokens[:, :seq_len]
            cond = c_stage1 if i < self.stage1_num_layers else c_stage2
            x = block(x, cond)

        return self.final_layer(x, c_stage2)
