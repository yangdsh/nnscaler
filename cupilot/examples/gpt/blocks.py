import torch
import cube


@cube.graph.parser.register('L^ N E^, (h+ g+ d+) E^, (h+ g+ d+), (h+ d+ 2) E^, (h+ d+ 2), E^ (h+ g+ d+) -> L^ N E^', name='self_attention_gqa')
def self_attention_gqa(query: torch.Tensor,
                            q_proj: torch.Tensor, q_bias: torch.Tensor,
                            kv_proj: torch.Tensor, kv_bias: torch.Tensor,
                            out_proj: torch.Tensor,
                            h: int, g: int, scale: float, dropout_p: float, mask: bool = False):
    q_heads = h * g
    L, N = query.size(0), query.size(1)
    q_dim = q_proj.size(0) // q_heads
    kv_heads = h
    d = kv_proj.size(0) // kv_heads // 2 # d == q_dim

    # Project the queries
    q = torch.nn.functional.linear(query, q_proj, q_bias)  # L N E, (g h d) E -> L N (g h d)
    q = q.view(L, N, q_heads, q_dim)  # L N (g h d) -> L N (g h) d
    q = q.permute(1, 2, 0, 3).contiguous().view(N * q_heads, L, q_dim)  # L N (g h) d -> (N g h) L d

    # Project the keys and values
    kv = torch.nn.functional.linear(query, kv_proj, kv_bias)  # L N E, (h d 2) E -> L N (h d 2)
    kv = kv.view(L, N, kv_heads, d * 2)  # L N (h d 2) -> L N h (d 2)
    k, v = kv.chunk(2, dim=-1)  # L N h (d 2) -> L N h d, L N h d
    k = k.permute(1, 2, 3, 0).contiguous().view(N * kv_heads, d, L)  # L N h d -> (N h) d L
    v = v.permute(1, 2, 0, 3).contiguous().view(N * kv_heads, L, d)  # L N h d -> (N h) L d

    # preallocating input tensor: (N g h) L L
    matmul_input_buffer = torch.empty([N * g * h, L, L], dtype=query.dtype, device=query.device)
    k_expanded = k.view(N, h, d, L).unsqueeze(1).expand(N, g, h, d, L).reshape(N * g * h, d, L)
    # print("self attn gqa", matmul_input_buffer.shape, q.shape, k_expanded.shape)
    attn = torch.baddbmm(
        matmul_input_buffer,
        q,  # (N g h) L d
        k_expanded, # (N g h) d L
        beta=0.0, alpha=scale
    )
    if mask:
        amask = torch.tril(torch.ones((L, L), device=query.device)).unsqueeze(0).unsqueeze(0)  # 1 1 L L
        attn = attn.masked_fill(amask == 0, float('-inf'))

    attn = torch.nn.functional.softmax(attn, dim=-1)  # (N g h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, training=True)

    # Compute the output
    v_expanded = v.view(N, h, L, d).unsqueeze(1).expand(N, g, h, L, d).reshape(N * g * h, L, d)
    output = torch.bmm(attn, v_expanded)  # (N g h) L L, (N g h) L d -> (N g h) L d
    output = output.view(N, g, kv_heads, L, d).permute(3, 0, 1, 2, 4).contiguous().view(L, N, g * kv_heads * d)  
    # (N g h) L d -> L N (g h d)
    output = torch.nn.functional.linear(output, out_proj)  # L N (g h d), E (g h d) -> L N E

    return output

class MultiHeadGroupedQueryAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, q_heads: int, inner_dim: int, group_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.kv_heads = q_heads // group_size
        self.inner_kv_dim = inner_dim // group_size
        self.q_heads = q_heads
        self.head_dim = inner_dim // q_heads
        self.group_size = group_size
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Query projection [(g h d), E]
        self.q_proj = torch.nn.Parameter(torch.empty(inner_dim, embed_dim))
        self.q_bias = torch.nn.Parameter(torch.empty(inner_dim))

        # Key/Value projection [(h d 2), E]
        self.kv_proj = torch.nn.Parameter(torch.empty(self.inner_kv_dim * 2, embed_dim))
        self.kv_bias = torch.nn.Parameter(torch.empty(self.inner_kv_dim * 2))

        # Output projection
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.q_proj)
        torch.nn.init.xavier_uniform_(self.kv_proj)
        torch.nn.init.xavier_uniform_(self.out_proj)
        torch.nn.init.zeros_(self.q_bias)
        torch.nn.init.zeros_(self.kv_bias)
        torch.nn.init.zeros_(self.out_bias)

    def forward(self, query):
        attn = self_attention_gqa(
            query, self.q_proj, self.q_bias,
            self.kv_proj, self.kv_bias,
            self.out_proj,
            self.q_heads // self.group_size, self.group_size, self.scaling, self.dropout_p, mask=False
        )
        attn = attn + self.out_bias
        return attn


@cube.graph.parser.register('L^ N E^, (h+ d^ 3) E^, (h+ d^ 3), E^ (h+ d^) -> L^ N E^', name='self_attention')
def self_attention(query: torch.Tensor, 
                   qkv_proj: torch.Tensor, qkv_bias: torch.Tensor,
                   out_proj: torch.Tensor,
                   h: int, scale: float, dropout_p: float, mask: bool = False):
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = qkv_proj.size(0) // num_head // 3

    qkv = torch.nn.functional.linear(query, qkv_proj, qkv_bias) # L N E, (h d 3) E -> L N (h d 3)
    qkv = qkv.view(L, N, num_head * dim_head, 3) # L N (h d 3) -> L N (h d) 3
    q, k, v = qkv.chunk(3, dim=-1)  # L N (3 h d) -> L N (h d), L N (h d), L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d

    # preallocating input tensor: (N h) L L
    matmul_input_buffer = torch.empty([N * h, L, L], dtype=query.dtype, device=query.device)
    # print("self attn", matmul_input_buffer.shape, q.shape, k.shape)
    # L (N h) d, L (N h) d -> (N h) L L
    attn = torch.baddbmm(
        matmul_input_buffer,
        q.transpose(0, 1),  # (N h) L d
        k.transpose(0, 1).transpose(1, 2), # (N h) d L
        beta=0.0, alpha=scale
    )
    # ======== replace the semantic into more efficient implementation ============

    # attention mask
    if mask: # (N h) L L -> (N h) L L
        attn = attn.view(N, num_head, L, L)
        ones = torch.ones((N, L, L), device=attn.device)
        amask = torch.tril(ones)
        amask = amask.view(N, 1, L, L)
        amask = (amask < 0.5)
        attn = attn.masked_fill_(amask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, True, False) # (N h) L L -> (N h) L L
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, out_proj) # L N (h d), E E  -> L N E
    return output


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # QKV [(h d 3), E]
        self.qkv_proj = torch.nn.Parameter(torch.empty(3 * inner_dim, embed_dim))
        self.qkv_bias = torch.nn.Parameter(torch.empty(3 * inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, query):
        attn = self_attention(
            query, self.qkv_proj, self.qkv_bias,
            self.out_proj,
            self.num_heads, self.scaling, self.dropout_p, mask=False
        )
        attn = attn + self.out_bias
        return attn


@cube.graph.parser.register('L^ N E, H+ E, H+, E H+ -> L^ N E', name='feedforward')
def feedforward(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor,
                dropout: float,
                is_training: bool = True) -> torch.Tensor:
    x = torch.nn.functional.linear(x, proj1, proj1_bias)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, dropout, is_training, False)
    x = torch.nn.functional.linear(x, proj2, None)
    return x


class MLP(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj1 = torch.nn.Parameter(torch.empty((hidden_dim, embed_dim)))
        self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim,)))
        self.proj2 = torch.nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = feedforward(x, self.proj1, self.proj1_bias,
                        self.proj2, self.dropout, self.training)
        x = x + self.proj2_bias
        return x


class TransformerLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int,
                 hidden_dropout: float = 0.0, attn_dropout: float = 0.0, activation_dropout: float = 0.0,
                 layernomr_eps: float = 1e-6):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, embed_dim, attn_dropout
        )
        self.self_attn_gqa = MultiHeadGroupedQueryAttention(
            embed_dim, num_heads, embed_dim
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim, eps=layernomr_eps)
        self.dropout = torch.nn.Dropout(p=hidden_dropout)
        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim, eps=layernomr_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        # x = self.self_attn_gqa(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        return x
