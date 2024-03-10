from lib.Modules import *
import timm
import torch
from thop import profile
'''
resnet50
'''


"""
消融实验 
"""


"""
slot
"""
class Network_interFA_noSpade_noEdge_ODE_slot_channel4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128, imagenet_pretrained=True):
        super(Network_interFA_noSpade_noEdge_ODE_slot_channel4, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.gcm_interFA = GCM_interFA_ODE_slot_channel4(256, channels)
        self.GPM = GPM()
        self.rem_decoder = REM_decoder_noSpade_noEdge(channels)  

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        f1, f2, f3, f4 = self.gcm_interFA(x1, x2, x3, x4)
        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        prior_cam = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        _, f4, f3, f2, f1 = self.rem_decoder([f1, f2, f3, f4], prior_cam, image)
        return prior_cam, f4, f3, f2, f1



class Network_interFA_noSpade_noEdge_ODE_slot_Channel4_Scribble(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128, imagenet_pretrained=True):
        super(Network_interFA_noSpade_noEdge_ODE_slot_Channel4_Scribble, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.gcm_interFA = GCM_interFA_ODE_slot_channel4_scribble(256, channels)
        self.GPM = GPM()
        self.rem_decoder = REM_decoder_noSpade_noEdge(channels)  

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        f1, f2, f3, f4, embed_feat = self.gcm_interFA(x1, x2, x3, x4)
        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        prior_cam = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        _, f4, f3, f2, f1 = self.rem_decoder([f1, f2, f3, f4], prior_cam, image)
        return prior_cam, f4, f3, f2, f1,embed_feat

if __name__ == '__main__':

    image = torch.rand(2, 3, 384, 384).cuda()
    model = Network_interFA_noSpade_noEdge_ODE_slot_Channel4_Scribble(64).cuda()
    pred_0, f4, f3, f2, f1,recon_combined31 = model(image)
    print(pred_0.shape)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(recon_combined31.shape)

    # net = model() 
    # img1 = torch.randn(1, 3, 512, 512)
    # img2 = torch.randn(1, 3, 512, 512)
    # img3 = torch.randn(1, 3, 512, 512)
    macs, params = profile(model, (image))
    print('flops: ', 2*macs, 'params: ', params)


    # print(recon_combined32.shape)
    # print(recon_combined41.shape)
    # print(recon_combined42.shape)