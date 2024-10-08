import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block

from common.nets.crosstransformer import CrossTransformer
from common.nets.layer import make_conv_layers
from mamba_ssm import Mamba

class Transformer(nn.Module):
    def __init__(self, in_chans=512, joint_num=21, depth=4, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, joint_num, in_chans))
        self.blocks = nn.ModuleList([
            Block(in_chans, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(depth)])
    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x

class FuseFormer(nn.Module):
    def __init__(self):
        super(FuseFormer, self).__init__()
        self.FC = nn.Linear(512*2, 512)
        self.pos_embed = nn.Parameter(torch.randn(1, 577, 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.SA_T = nn.ModuleList([
            Block(512, 4, 4.0, qkv_bias=False, norm_layer=nn.LayerNorm)
            for i in range(1)])
        self.FC2 = nn.Linear(512, 512)
        #Decoder
        self.CA_T = CrossTransformer()
        self.FC3 = nn.Linear(512, 512)

        self.mamba = Mamba(
            d_model=512,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2  # Block expansion factor
        )

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        feat1 = rearrange(feat1, 'B C H W -> B (H W) C')
        feat2 = rearrange(feat2, 'B C H W -> B (H W) C')
        # joint Token
        token_j = self.FC(torch.cat((feat1, feat2), dim=-1))
        
        # similar token
        token_s = torch.cat((feat1, feat2), dim=1)
        token_s = token_s + + self.pos_embed[:,1:]
        cls_token = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        token_s = torch.cat((cls_token, token_s), dim=1)
        # token_s = self.mamba(token_s)
        for blk in self.SA_T:
            token_s = blk(token_s)
        token_s = self.FC2(token_s)

        output = self.CA_T(token_j, token_s)

        output = self.FC3(output)
        output = rearrange(output, 'B (H W) C -> B C H W', H=H, W=W)
        return output



class EABlock(nn.Module):
    def __init__(self):
        super(EABlock, self).__init__()
        self.conv_l = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.conv_r = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.Extract = FuseFormer()
        self.Adapt_r = FuseFormer()
        self.Adapt_l = FuseFormer()
        self.conv_l2 = make_conv_layers([512*2, 512*2], kernel=1, stride=1, padding=0)
        self.conv_r2 = make_conv_layers([512*2, 512*2], kernel=1, stride=1, padding=0)

        self.conv_fusion = make_conv_layers([512*4, 512*4], kernel=1, stride=1, padding=0)

    def forward(self, hand_feat):
        rhand_feat = self.conv_r(hand_feat)
        lhand_feat = self.conv_l(hand_feat)
        inter_feat = self.Extract(rhand_feat, lhand_feat)

        rinter_feat = self.Adapt_r(rhand_feat, inter_feat)
        linter_feat = self.Adapt_l(lhand_feat, inter_feat)

        rhand_feat = self.conv_r2(torch.cat((rhand_feat,rinter_feat),dim=1))
        lhand_feat = self.conv_l2(torch.cat((lhand_feat,linter_feat),dim=1))

        output = self.conv_fusion(torch.cat((rhand_feat, lhand_feat), dim=1))
        return output





