import torch
import torch.nn as nn
from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D
import torch.nn.functional as F


class TreeEnergyLoss(nn.Module):
    def __init__(self, configer=None):
        super(TreeEnergyLoss, self).__init__()
        self.configer = configer
        if self.configer is None:
            print("self.configer is None")

        self.weight = 0.4
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.002)

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            N = unlabeled_ROIs.sum()

        prob = torch.softmax(preds, dim=1)

        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]
        print(AS)
        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N

        return self.weight * tree_loss

class TreeEnergyLossBinary(nn.Module):
    def __init__(self, configer=None):
        super(TreeEnergyLossBinary, self).__init__()
        self.configer = configer
        if self.configer is None:
            print("self.configer is None")

        self.weight = 0.4
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.002)

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            N = unlabeled_ROIs.sum()

        # prob = torch.softmax(preds, dim=1)

        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=preds, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(preds - AS)).sum()
        if N > 0:
            tree_loss /= N

        return self.weight * tree_loss
    
class TreeEnergyLossBinarySAM(nn.Module):
    #TODO
    def __init__(self, configer=None):
        super(TreeEnergyLossBinarySAM, self).__init__()
        self.configer = configer
        if self.configer is None:
            print("self.configer is None")

        self.weight = 0.4
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.002)

    def forward(self, preds, low_feats, high_feats,SAMSegment):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            SAMSegment = F.interpolate(SAMSegment.float(), size=(h, w), mode='nearest')

        # prob = torch.softmax(preds, dim=1)

        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=preds, embed_in=low_feats, tree=tree)  # [b, n, h, w]


        SAMforeground = torch.sigmoid(SAMSegment)
        SAMbackground = 1 - SAMforeground
        # SAMPred = torch.cat([SAMforeground, SAMbackground], 1)

        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        AS = (SAMbackground+AS)/2
        # unlabeled_ROIs = 
        # N = unlabeled_ROIs.sum()
        tree_loss = (unlabeled_ROIs * torch.abs(preds - AS)).sum()
        if N > 0:
            tree_loss /= N

        return self.weight * tree_loss


if __name__ == '__main__':
    loss = TreeEnergyLoss().cuda()
    preds = torch.rand(4, 2, 96, 96).cuda()
    low_feats = torch.rand(4, 3, 384, 384).cuda()
    high_feats = torch.rand(4, 256, 96, 96).cuda()
    unlabeled_ROIs = torch.rand(4, 384, 384).cuda()
    loss1 = loss(preds, low_feats, high_feats, unlabeled_ROIs)
    print(loss1)
