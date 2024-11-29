from ...imports import *


class PositionalEmbedding(nn.Module):

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5, trainable: bool = True):
        super().__init__()
        self.input_shape = input_shape
        self.emb = nn.Parameter(torch.zeros(1, *input_shape), requires_grad=trainable)
        self.use_dropout = dropout_rate is not None and dropout_rate != 0.
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.emb
        if self.use_dropout:
            x = self.dropout(x)
        return x

    @property
    def trainable(self):
        return self.emb.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.emb.requires_grad = value


class SinCosPositionalEmbedding(PositionalEmbedding):

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__(input_shape, dropout_rate, trainable=False)
        self.emb.data = self.make_embedding().unsqueeze(0)

    def make_embedding(self) -> Tensor:
        n_position, d_hid = self.input_shape

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2 * torch.div(torch.arange(d_hid), 2, rounding_mode='trunc') / d_hid)

        sinusoid_table = torch.stack([get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.float()
    
class AudioPatchEmbed(nn.Module):
    """ Audio to Patch Embedding"""

    def __init__(
        self,
        img_size=173,
        patch_size=[16, 16],
        in_chans=1,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedding3d(nn.Module):

    def __init__(self, input_size: Shape, patch_size: Union[int, Shape], embedding: int,
        strides: Optional[Union[int, Shape]] = None,
        build_normalization: Optional[ModuleFactory] = None
    ):
        super().__init__()
        # channel, time, height, width
        c, t, h, w = input_size
        # patch_time, patch_height, patch_width
        pt, ph, pw = (patch_size, patch_size, patch_size) if type(patch_size) is int else patch_size

        # configure the strides for conv3d
        if strides is None:
            # no specified means no overlap and gap between patches
            strides = (pt, ph, pw)
        elif type(strides) is int:
            # transform the side length of strides to 3D
            strides = (strides, strides, strides)

        self.projection = nn.Conv3d(c, embedding, kernel_size=(pt, ph, pw), stride=strides)
        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
        self.rearrange = Rearrange("b d nt nh nw -> b (nt nh nw) d")

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        # print('Aftre projection:', x.shape)
        x = self.rearrange(x)
        # print('After rearrange:', x.shape)
        if self.has_norm:
            x = self.normalization(x)
        return x
    
class SimplePatchify2D(nn.Module):
    def __init__(self, H, W, h, w):
        super().__init__()
        self.H = H
        self.W = W
        self.h = h
        self.w = w

    def forward(self, x):
        # Input x is of shape [batch_size, 1, H, W]
        x = x.unfold(2, self.h, self.h).unfold(3, self.w, self.w)
        # After unfolding, x is of shape [batch_size, 1, num_height_patches, num_width_patches, patch_height, patch_width]
        x = rearrange(x, "b c nh nw ph pw -> b (nh nw) (c ph pw)")
        return x

    def unpatch(self, x):
        # Reshape the patches back into the unfolded form
        x = rearrange(x, "b (nh nw) (c ph pw) -> b c nh nw ph pw",
                      nh=self.H // self.h, nw=self.W // self.w,
                      ph=self.h, pw=self.w)

        # Merge the patches back together
        x = rearrange(x, "b c nh nw ph pw -> b c (nh ph) (nw pw)")
        return x
    

class SimplePatchify3D(nn.Module):
    def __init__(self, T, H, W, h, w, t):
        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.h = h
        self.w = w
        self.t = t

    def forward(self, x):
        x = x.unfold(2, self.t, self.t).unfold(3, self.h, self.h).unfold(4, self.w, self.w)
        x = rearrange(x, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
        return x

    def unpatch(self, x):
        # Reshape the patches back into the unfolded form
        x = rearrange(x, "b (nt nh nw) (c pt ph pw) -> b c nt nh nw pt ph pw",
                      nt=self.T // self.t, nh=self.H // self.h, nw=self.W // self.w,
                      pt=self.t, ph=self.h, pw=self.w)

        # Merge the patches back together
        x = rearrange(x, "b c nt nh nw pt ph pw -> b c (nt pt) (nh ph) (nw pw)")
        return x
    
    def unit_test(self):
        z = torch.randn(1, 3, self.T, self.H, self.W)
        x = self(z)
        z_recon = self.unpatch(x)
        assert z.shape == z_recon.shape, f"Shape mismatch: {z.shape} vs {z_recon.shape}"
        err = (z - z_recon).abs()
        print(err.mean(), err.std(), err.max(), err.min())