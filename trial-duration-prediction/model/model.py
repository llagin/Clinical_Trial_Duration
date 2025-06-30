import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Norm(nn.Module):
    """Layer normalization."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x):
        return self.norm(x)

class Residual(nn.Module):
    """Residual connection."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class MLP(nn.Module):
    """Feedforward network."""
    def __init__(self, hidden_dim, intermediate_dim, activation_fn=nn.GELU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            activation_fn,
            nn.Linear(intermediate_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_length, dim = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        q, k, v = qkv
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out(out)

class Transformer(nn.Module):
    """Transformer Encoder."""
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleList([
                Norm(hidden_size),
                Norm(hidden_size),
                Attention(hidden_size, num_attention_heads),
                MLP(hidden_size, intermediate_size)
            ]) for _ in range(num_hidden_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            norm1, norm2, attention, mlp = layer
            x = x + norm1(attention(x))
            x = x + norm2(mlp(x))
        return x

class FusionNet(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_size=100, num_heads=10):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.model_embeds = nn.ModuleList([
            nn.Linear(dim, hidden_size) for dim in input_dims
        ])

        self.model_transformer = nn.ModuleList([
            Transformer(hidden_size=self.hidden_size, num_hidden_layers=2,
                        num_attention_heads=self.num_heads, intermediate_size=self.hidden_size * 4) for _ in input_dims
        ])

        self.cross_transformer = Transformer(hidden_size=self.hidden_size, num_hidden_layers=12,
                                             num_attention_heads=self.num_heads, intermediate_size=self.hidden_size * 4)

        self.weight = nn.Parameter(torch.ones(len(input_dims)))
        self.fc = nn.Linear(hidden_size * len(self.input_dims), output_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size*len(self.input_dims), hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_size, output_dim)
        # )
        # self.fc = nn.Linear(hidden_size,output_dim)

    def forward(self, *args):
        embeddings = []
        for i, feat in enumerate(args):
            emb = self.model_embeds[i](feat)  # (batch, embed_dim)
            # emb = self.model_transformer[i](emb.unsqueeze(1))
            embeddings.append(emb.unsqueeze(1))  # (batch, 1, embed_dim)

        embeddings = torch.cat(embeddings, dim=1)  # (batch, seq_len, embed_dim)

        output = self.cross_transformer(embeddings)
        output = output.reshape(output.size(0), -1)  # (batch, seq_len * output_dim)
        # output = output.mean(dim=1)  # (batch, seq_len * output_dim)
        # weights = torch.softmax(self.weight, dim=0)  # (seq_len,)
        # output = (output * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        output = self.fc(output)

        return output

class ShortFusionNet(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_size=5, num_heads=1):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.model_embeds = nn.ModuleList([
            nn.Linear(dim, hidden_size) for dim in input_dims
        ])
        self.cross_transformer = Transformer(hidden_size=self.hidden_size, num_hidden_layers=4,
                                             num_attention_heads=self.num_heads, intermediate_size=self.hidden_size * 4)

        self.fc = nn.Linear(hidden_size * len(self.input_dims), output_dim)

    def forward(self, *args):
        embeddings = []
        for i, feat in enumerate(args):
            emb = self.model_embeds[i](feat)  # (batch, hidden_size)
            embeddings.append(emb.unsqueeze(1))  # (batch, 1, hidden_size)

        embeddings = torch.cat(embeddings, dim=1)  # (batch, seq_len, hidden_size)

        output = self.cross_transformer(embeddings)
        output = output.reshape(output.size(0), -1)  # (batch, seq_len * hidden_size)

        output = self.fc(output)

        return output
    
class Protocol_Attention_Regression_FACT_new(nn.Module):
    def __init__(self, output_dim=1):

        super(Protocol_Attention_Regression_FACT_new, self).__init__()
        self.short_input_dims = [4,5,4,1,1,11]
        self.short_input_dims_2 = [50, 50]
        self.input_dims = [768] * 7 + [30, 30]
        self.output_dim = output_dim
        self.short_fusion = ShortFusionNet(self.short_input_dims, 30)
        self.short_fusion_2 = ShortFusionNet(self.short_input_dims_2, 30)
        self.fusion = FusionNet(self.input_dims, 1)
        self.fc = nn.Sequential(
            nn.Linear(10 * 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, *args):
        if len(args) == 14:
            criteria_emb = args[0]
            drug_emb = args[1]
            disease_emb = args[2]
            phase_emb = args[3]
            title_emb = args[4]
            summary_emb = args[5]
            primary_purpose_emb = args[6]
            time_frame_emb = args[7]
            intervention_model_emb = args[8]
            masking_emb = args[9]
            enrollment_emb = args[10]
            location_emb = args[11]
            smiles_emb = args[12]
            icd_emb = args[13]
        
        else:
            raise ValueError("Invalid input format.")

        short_emb = self.short_fusion(phase_emb, intervention_model_emb, masking_emb, enrollment_emb,
                               location_emb, primary_purpose_emb)
        smiles_icd_emb = self.short_fusion_2(smiles_emb, icd_emb)
        split_emb = torch.split(criteria_emb, 768, dim=1)
        inclusion_emb = split_emb[0]
        exclusion_emb = split_emb[1]
        output = self.fusion(inclusion_emb, exclusion_emb, drug_emb, disease_emb, title_emb, summary_emb,
                             time_frame_emb, short_emb, smiles_icd_emb)
        return output