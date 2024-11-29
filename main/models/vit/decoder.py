from ...imports import *
from .encoder import VitEncoder
from .embeddings import SinCosPositionalEmbedding

class ViTDecoder(VitEncoder):
    def __init__(self, audio_num_patches, video_num_patches,
                   audio_target_embed_dim, video_target_embed_dim,
                     embed_dim=384, depth=4,
        num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0.):
        self.audio_num_patches = audio_num_patches
        self.video_num_patches = video_num_patches
        super().__init__(
            img_size=None, patch_size=None,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, init_values=init_values
        )


        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim))
        
        self.video_proj = nn.Linear(embed_dim, video_target_embed_dim)
        
    def init_positional_embedding(self, num_patches, embed_dim, dropout_rate=0):
        self.video_pos_embedding = SinCosPositionalEmbedding((self.video_num_patches, embed_dim), dropout_rate=dropout_rate)
    
    def init_patch_embedding(self, img_size, n_frames, patch_size, tubelet_size, embed_dim):
        self.num_patches = None
        pass # We dont need patch embedding in decoder

    def cat_mask(self, mask_token, embeddings, ids_restore):
        mask_tokens = mask_token.repeat(
            embeddings.shape[0], ids_restore.shape[1] - embeddings.shape[1], 1)
        x_ = torch.cat([embeddings, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, embeddings.shape[2]))  # unshuffle
        return x_
    
    def forward(self,video_embeddings):
        
        video_embeddings = self.video_pos_embedding(video_embeddings)
        video_embeddings = self.backbone(video_embeddings)
        video_embeddings = torch.sigmoid(video_embeddings)
        return video_embeddings
