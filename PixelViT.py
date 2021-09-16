import torch
import torch.nn as nn
import torch.utils.checkpoint
from functools import partial
from gpt2.utils.fusing import LayerNorm
from gpt2.modeling import (PadMasking, FutureMasking, AttentionLayer, Past,
                           PositionalEmbedding, TokenEmbedding,
                           PositionwiseFeedForward)

from typing import Optional, Tuple, List, Union

import matplotlib.pyplot as plt

from einops.layers.torch import Rearrange

class TransformerLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., seq_len, past_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., past_len + seq_len, dims)
    ===========================================================================
    """
    def __init__(self,
                 heads: int,
                 dims: int,
                 rate: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, dropout)
        self.ff = PositionwiseFeedForward(dims, rate, dropout)
        self.ln_attn = LayerNorm(dims)
        self.ln_ff = LayerNorm(dims)

    def forward(self,
                x: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        # Layer normalizations are performed before the layers respectively.
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)
        
        x = x + a
        x = x + self.ff(self.ln_ff(x))

        return x if self.training else (x, past)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Transformer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               long            (..., seq_len)
    past (**)       float           (..., past_len, dims)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (**)   float           (..., past_len + seq_len, dims)
    ===========================================================================
    """
    def __init__(self,
                 channels: int,
                 patch_size: int,
                 image_size: int,
                 dim: int,
                 layers: int,
                 pad_idx: int,
                 words: int,
                 seq_len: int,
                 heads: int,
                 rate: int = 4,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 num_classes = 30):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2

        patch_dim = channels * patch_size ** 2
        patch_height, patch_width = pair(patch_size)
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()

        self.mask = torch.triu(torch.ones(1, num_patches, num_patches), diagonal=1)
        #self.mask = (self.mask == 0)
        self.mask = self.mask.bool()


        #self.positional_embedding = PositionalEmbedding(seq_len, dim)
        self.token_embedding = TokenEmbedding(words, dim)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList([
            TransformerLayer(heads, dim, rate, dropout)
            for _ in range(layers)])

        self.ln_up = LayerNorm(dim)
        self.upsample = nn.Linear(dim, dim*4, bias=False)

        self.from_patch = nn.Sequential(
            Rearrange('b c (p1 p2) -> b c (p1) (p2)', p1 = 16, p2 = dim//4),
            Rearrange('b c (p1 p2) l -> b c (p1) (p2) l', p1 = 4, p2 = 4)
        )

        self.to_img = nn.Sequential(
            Rearrange('b (p1 p2) h w l -> b (p1 h) (p2 w) l', p1 = 8, p2 = 8)
        )

        self.ln_head = LayerNorm(dim//4)
        self.head = nn.Linear(dim//4, num_classes, bias=False)


    def forward(self,
                x: torch.Tensor,
                past: Optional[List[Past]] = None,
                use_grad_ckpt: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
        b = x.size(0)

        #offset = past[0][0].size(-2) if past is not None else 0
        #print(x.size())
        x = self.to_patch_embedding(x)
        
        #print('Pat:', x.size())
        #print('Pos:', self.pos_embedding.size())
        x = x + self.pos_embedding
        
        # Use token embedding and positional embedding layers.
        #x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)
        # Apply transformer layers sequentially.
        present = []

        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint, transformer)

            x = transformer(x, past[i] if past is not None else None, self.mask)

            if not self.training:
                present.append(x[1])
                x = x[0]

        x = self.ln_up(x)
        x = self.upsample(x)

        #print(x.size())

        x = self.from_patch(x)
        x = self.to_img(x)

        #print(x.size())

        x = self.ln_head(x)
        #x = self.token_embedding(x, transposed=True)

        #print(x.size())

        #x = x if self.training else (x, present)
        logits = self.head(x)        
        logits = logits.view(b, -1, 30)
        return logits


if __name__ == '__main__':
    import os
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,7'
    #print('Visible Device:', os.environ['CUDA_VISIBLE_DEVICES'])
    #import numpy as np
    #import cv2
    model = Transformer( channels = 1,
                         patch_size = 4,
                         image_size = 32,
                         dim = 512,            # embed_dim
                         heads = 8,            # num_heads
                         layers = 16,           # num_layers
                         words = 256,           # num_vocab
                         seq_len = 8*8,      # num_positions
                         bidirectional = False,
                         dropout = 0.1,
                         rate   = 4,
                         pad_idx = 1,
                         num_classes = 30)
    
    x = torch.zeros((8, 1, 32, 32)).type(torch.float32)
    #x = torch.zeros((8, 1024)).type(torch.float32)
    #x = cv2.imread(r'C:\Users\william\ImageGPT\plaid.png', 0)
    #x = np.expand_dims(x, 0)
    #x = np.expand_dims(x, 0)
    #x = torch.from_numpy(x)
    #x = x.to(torch.float32)
    #x /= 255
    logits = model(x)
    print('Logits:', logits.size())

    #plt.imshow(logits)
    
    
    
    
    
