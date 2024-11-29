from ...imports import *


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
        build_activation: Optional[ModuleFactory] = None,
        build_normalization: Optional[ModuleFactory] = None,
        normalization_after_activation: bool = False,
        dropout_rate: float = 0.
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        self.has_act = build_activation is not None
        if self.has_act:
            self.activation = build_activation()
        else:
            self.activation = None

        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
            self.norm_after_act = normalization_after_activation
        else:
            self.normalization = None

        self.has_dropout = dropout_rate > 0
        if self.has_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.has_act and self.has_norm:
            if self.norm_after_act:
                x = self.activation(x)
                x = self.normalization(x)
            else:
                x = self.normalization(x)
                x = self.activation(x)
        elif self.has_act and not self.has_norm:
            x = self.activation(x)
        elif not self.has_act and self.has_norm:
            x = self.normalization(x)

        if self.has_dropout:
            x = self.dropout(x)
        return x


class MLP(nn.Module):

    def __init__(self, neurons: Sequence[int],
        build_activation: Optional[ModuleFactory] = None, dropout_rate: float = 0.
    ):
        super().__init__()
        n_features = neurons[1:]
        self.layers: nn.ModuleList[Linear] = nn.ModuleList(
            [Linear(neurons[i], neurons[i + 1], True, build_activation, None,
                False, dropout_rate
            ) for i in range(len(n_features) - 1)
            ] + [
                Linear(neurons[-2], neurons[-1], True)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
        proj_drop=0., attn_head_dim=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # The original qkv layer for self-attention
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_map = None  # Store attention maps for visualization
    def forward(self, x1, x2=None):
        """
        If only x1 is provided, do traditional self-attention.
        If both x1 and x2 are provided, do cross-attention where:
        x1: tensor for key and value (shape: B, N, C)
        x2: tensor for query (shape: B, N, C)
        """
        if x2 is None:
            # Traditional self-attention
            B, N, C = x1.shape
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            # Compute q, k, v from x1
            qkv = F.linear(input=x1, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Split into query, key, value
        else:
            # Cross-attention: key and value come from x1, query comes from x2
            B, N, C = x2.shape  # Query shape determines batch size, sequence length, and channel dimension
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            # Query (q) from x2
            q = F.linear(input=x2, weight=self.qkv.weight[:C, :], bias=qkv_bias[:C])
            q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # Split q into heads
            
            # Key (k) and Value (v) from x1
            kv = F.linear(input=x1, weight=self.qkv.weight[C:, :], bias=qkv_bias[C:])
            kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # Split key and value
        # Scale the query
        q = q * self.scale
        # Compute attention scores (q @ k^T)
        attn = (q @ k.transpose(-2, -1))
        # Apply softmax to attention scores
        attn = attn.softmax(dim=-1)
        self.attn_map = attn  # Store the attention map for visualization
        # Apply dropout to attention
        attn = self.attn_drop(attn)
        # Compute the final attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # Apply final projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            neurons=[dim, mlp_hidden_dim, dim],
            build_activation=act_layer,
            dropout_rate=drop
        )

        self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)  
        self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)  

    
    def forward(self, x1, x2=None):
        """
        If x2 is None, perform self-attention.
        If both x1 and x2 are provided, perform cross-attention.
        """
        if x2 is None:

            # Standard self-attention path
            x1 = x1 + self.attn(self.norm1(x1))
            x1 = x1 + self.mlp(self.norm2(x1))
        else:
 
            # Standard cross-attention path
            x1 = x1 + self.attn(self.gamma1*self.norm1(x1), self.gamma2*self.norm1(x2))  # Pass both x1 and x2 to attn
            x1 = x1 + self.mlp(self.norm2(x1))
        return x1
