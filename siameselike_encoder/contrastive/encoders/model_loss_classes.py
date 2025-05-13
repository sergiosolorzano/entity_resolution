import os, sys
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.logger import Tne_Logger

import config

class TailSuppressedTripletMarginLoss(nn.Module):
    def __init__(self, logger: 'Tne_Logger', device, output_feature_dim, margin=1, pct_p=0.01, pct_n=0.05):
        super(TailSuppressedTripletMarginLoss, self).__init__()
        self.feature_dim = output_feature_dim
        self.pct_p = pct_p # trains 1-pct_p, i.e. exclude easiest:dist_ap >= pos_threshold
        self.pct_n = pct_n #train bottom pct_n, i.e. hardest negs:dist_an <= neg_threshold
        self.margin = margin
        self.triplet_criterion = nn.TripletMarginLoss(self.margin, p=2, reduction='none')
        self.logger = logger
        self.device = device

    def forward(self, anchor, positive, negative, epoch):
        
        dist_ap = torch.norm(anchor - positive, p=2, dim=1)
        dist_an = torch.norm(anchor - negative, p=2, dim=1)

        pos_threshold = torch.quantile(dist_ap, self.pct_p)
        neg_threshold = torch.quantile(dist_an, self.pct_n)

        # self.logger.info("\n")
        # self.logger.info(f"pos_threshold {pos_threshold.mean()} neg_threshold {neg_threshold.mean()}")
        # self.logger.info(f"dist_an {dist_an.mean()} dist_ap {dist_ap.mean()}")
        # self.logger.info(f"self.margin {self.margin}")
        
        #identify valid samples
        valid_positives = dist_ap >= pos_threshold
        valid_negatives = dist_an <= neg_threshold
        valid_triplets = valid_positives & valid_negatives

        valid_pos_count = valid_positives.sum().item()
        valid_neg_count = valid_negatives.sum().item()
        valid_trip_count = valid_triplets.sum().item()
        total_samples = anchor.size(0) #total tirplets

        self.logger.info(f"Batch valid positives: {valid_pos_count}/{total_samples} ({100.0 * valid_pos_count/total_samples:.2f}%)")
        self.logger.info(f"Batch valid negatives: {valid_neg_count}/{total_samples} ({100.0 * valid_neg_count/total_samples:.2f}%)")
        self.logger.info(f"Batch valid triplets: {valid_trip_count}/{total_samples} ({100.0 * valid_trip_count/total_samples:.2f}%)")

        #triplet loss
        all_losses = self.triplet_criterion(anchor, positive, negative)
        suppressed_loss = all_losses * valid_triplets.float()
        loss_mean = suppressed_loss.sum() / (valid_triplets.sum() + 1e-8)
        
        return loss_mean, valid_pos_count, valid_neg_count, valid_trip_count, total_samples, dist_ap, dist_an, pos_threshold, neg_threshold

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        
        distances = torch.norm(embedding1 - embedding2, p=2, dim=1)
        #loss implementation https://www.researchgate.net/publication/4246277_Dimensionality_Reduction_by_Learning_an_Invariant_Mapping
        # print("ap 1-labvel",(1-label),"ap distances",distances**2, "result ap",0.5 * ((1-label) * distances**2))
        # print("an label",label,"an distances",distances, "margin",self.margin, "self.margin - distances",self.margin - distances,"result an", torch.clamp(self.margin - distances, min=0)**2)
        loss = 0.5 * ((1-label) * distances**2 + label * torch.clamp(self.margin - distances, min=0)**2)
        return loss.mean(), distances