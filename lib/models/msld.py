import math
import random

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34

from nms import nms
from lib.lane import Lane
from lib.focal_loss import FocalLoss

from .layer import MultiSpectralAttentionLayer, SingleMultiSpectralAttentionLayer
from .resnet import resnet122 as resnet122_cifar
from .matching import match_proposals_with_targets


class MSLD(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 pretrained_backbone=True,
                 S=72,
                 img_w=640,
                 img_h=360,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 anchor_feat_channels=64):
        super(MSLD, self).__init__()
        # Some definitions
        # self.feature_extractor, backbone_nb_channels, self.stride = build_simple_resnet([8, 16, 32, 64], [1,2,2,2], [2,2,2,2])
        self.feature_extractor, backbone_nb_channels, self.stride = get_backbone(backbone, pretrained_backbone)
        self.img_w = img_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = math.ceil(img_h / self.stride)
        fmap_w = math.ceil(img_w / self.stride)
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]

        # Pre compute indices for the anchor pooling
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.anchor_feat_channels*self.fmap_h, nhead=2)
        # self.transformer_model = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # hidden_dim = 128
        # dim_feedforward = 256
        # pos_type = 'sine'
        # num_queries = 7
        # drop_out = 0.1
        # num_heads = 2
        # enc_layers = 2
        # dec_layers = 2
        # pre_norm = False
        # return_intermediate = False 
        

        # self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.input_proj = nn.Conv2d(self.anchor_feat_channels * self.fmap_h, hidden_dim, kernel_size=1)  # the same as channel of self.layer4

        # self.transformer = build_transformer(hidden_dim=hidden_dim,
        #                                      dropout=drop_out,
        #                                      nheads=num_heads,
        #                                      dim_feedforward=dim_feedforward,
        #                                      enc_layers=enc_layers,
        #                                      dec_layers=dec_layers,
        #                                      pre_norm=pre_norm,
        #                                      return_intermediate_dec=return_intermediate)

        # Setup and initialize layers

        self.att_layer = MultiSpectralAttentionLayer(backbone_nb_channels, dct_h=7, dct_w=7)
        self.n_att_layer = SingleMultiSpectralAttentionLayer(self.anchor_feat_channels, dct_n=8)
        # self.conv1 = nn.Conv2d(backbone_nb_channels, self.anchor_feat_channels, kernel_size=1)
        # group conv
        # self.backbone_nb_channels = backbone_nb_channels
        # self.group_channels = backbone_nb_channels // 3
        # self.extra_channels = backbone_nb_channels % 3
        # self.g_conv1 = nn.Conv2d(self.group_channels, self.anchor_feat_channels, kernel_size=1)
        # self.g_conv2 = nn.Conv2d(self.group_channels, self.anchor_feat_channels, kernel_size=1)
        # self.g_conv3 = nn.Conv2d(self.group_channels + self.extra_channels, self.anchor_feat_channels, kernel_size=1)
        # offset layer 
        # self.conv2 = nn.Conv2d(self.anchor_feat_channels, 1, kernel_size=1)
        self.cls_layer = nn.Linear(2*self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer = nn.Linear(2*self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        # self.initialize_layer(self.g_conv1)
        # self.initialize_layer(self.g_conv2)
        # self.initialize_layer(self.g_conv3)
        # self.initialize_layer(self.conv2)
        # self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def forward(self, x, conf_threshold=None, nms_thres=0, nms_topk=3000):
        batch_features = self.feature_extractor(x)
        # print('shape:', batch_features.shape)

        # Feature Decouping and Compression Procedure
        # Group Process 1. Block Split; 2. One Interval; 3. Random
        # 1. Block Split
        # batch_g1_features = batch_features[:, :self.group_channels, ::]
        # batch_g2_features = batch_features[:, self.group_channels:2*self.group_channels, ::]
        # batch_g3_features = batch_features[:, 2*self.group_channels:, ::]
        # 2. One Interval
        # g1_ind = torch.arange(0, self.backbone_nb_channels-2, 3)
        # g2_ind = torch.arange(1, self.backbone_nb_channels-1, 3)
        # g3_ind = torch.zeros([self.group_channels+self.extra_channels])
        # g3_ind[:self.group_channels] = torch.arange(2, self.backbone_nb_channels, 3)
        # g3_ind[-2] = self.backbone_nb_channels-2
        # g3_ind[-1] = self.backbone_nb_channels-1
        # batch_g1_features = batch_features[:, g1_ind.long(), ::]
        # batch_g2_features = batch_features[:, g2_ind.long(), ::]
        # batch_g3_features = batch_features[:, g3_ind.long(), ::]
        # # 3. Random
        # random.seed(255)
        # random_ind = torch.arange(0, self.backbone_nb_channels, 1)
        # random.shuffle(random_ind)
        # g1_ind = random_ind[:self.group_channels]
        # g2_ind = random_ind[self.group_channels:2*self.group_channels]
        # g3_ind = random_ind[2*self.group_channels:]
        # batch_g1_features = batch_features[:, g1_ind.long(), ::]
        # batch_g2_features = batch_features[:, g2_ind.long(), ::]
        # batch_g3_features = batch_features[:, g3_ind.long(), ::]

        # s = self.g_conv1(batch_g1_features)
        # z = self.g_conv2(batch_g2_features)
        # u = self.g_conv3(batch_g3_features)
        # mean_z = z.mean(1).unsqueeze(1)
        # std_z = z.std(1).unsqueeze(1)
        # batch_features = s * (z - mean_z) / std_z + u
        batch_features = self.att_layer(batch_features)
        # batch_features = self.conv1(batch_features)

        batch_anchor_features = self.cut_anchor_features(batch_features)
        batch_anchor_features = batch_anchor_features.view(-1, *batch_anchor_features.shape[2:4])
        batch_anchor_features = self.n_att_layer(batch_anchor_features)
        # # Feature Selecting Procedure
        # m_batch_anchor_features = batch_anchor_features.squeeze(4).permute(0, 2, 1, 3)
        # cut_batch_offset = self.conv2(m_batch_anchor_features).permute(0, 2, 3, 1)
        # m_batch_offset = torch.repeat_interleave(cut_batch_offset, self.anchor_feat_channels, dim=1).reshape(cut_batch_offset.shape[0], -1, 1)
        # # define range e.g., (-3, 3)
        # m_batch_offset = (torch.sigmoid(m_batch_offset) - 0.5) * 2
        # # cut_batch_offset = (torch.sigmoid(cut_batch_offset) - 0.5) * 3 * self.stride
        # # batch_offset = nn.functional.interpolate(cut_batch_offset.squeeze(3), size=[self.n_offsets], mode='linear')
        # batch_anchor_features = self.cut_offset_anchor_features(m_batch_offset, batch_features)

        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)
        # batch_anchor_features = batch_anchor_features.view(*batch_anchor_features.shape[:2], self.anchor_feat_channels * self.fmap_h).permute(1,0,2)
        # transformer_features = self.transformer_model(batch_anchor_features)
        # batch_anchor_features = batch_anchor_features.squeeze(4)
        # masks = torch.zeros(batch_anchor_features.shape[0], 1, *batch_anchor_features.shape[2:]).cuda()
        # pmasks = F.interpolate(masks[:, 0, :, :][None], size=batch_anchor_features.shape[-2:]).to(torch.bool)[0]
        # pos    = self.position_embedding(batch_anchor_features, pmasks)
        # print(pos.shape)
        # print(batch_anchor_features.shape)
        # batch_anchor_features, _, weights  = self.transformer(batch_anchor_features, pmasks, self.query_embed.weight, pos)

        # Add attention features
        # softmax = nn.Softmax(dim=2)
        # scores = self.attention_layer(batch_anchor_features)
        # attention = softmax(scores)
        # attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        # non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        # attention_matrix[:] = 0
        # attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        # batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1)
        # attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
        #                                torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        # batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=2)

        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1)
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device)
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg

        # Apply nms
        proposals_list = self.nms(reg_proposals, nms_thres, nms_topk, conf_threshold)
        return proposals_list

    # def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold):
    #     softmax = nn.Softmax(dim=1)
    #     proposals_list = []
    #     for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
    #         anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
    #         # The gradients do not have to (and can't) be calculated for the NMS procedure
    #         with torch.no_grad():
    #             scores = softmax(proposals[:, :2])[:, 1]
    #             if conf_threshold is not None:
    #                 # apply confidence threshold
    #                 above_threshold = scores > conf_threshold
    #                 proposals = proposals[above_threshold]
    #                 scores = scores[above_threshold]
    #                 anchor_inds = anchor_inds[above_threshold]
    #             if proposals.shape[0] == 0:
    #                 proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
    #                 continue
    #             keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
    #             keep = keep[:num_to_keep]
    #         proposals = proposals[keep]
    #         anchor_inds = anchor_inds[keep]
    #         attention_matrix = attention_matrix[anchor_inds]
    #         proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

    #     return proposals_list
    def nms(self, batch_proposals, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals in batch_proposals:
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            proposals_list.append((proposals, self.anchors[keep], anchor_inds))

        return proposals_list

    def loss(self, proposals_list, targets, cls_loss_weight=10):
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors, _), target in zip(proposals_list, targets):
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = match_proposals_with_targets(
                    self, anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # Regression targets
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts
                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = (positive_starts + target[:, 4] - 1).round().long()
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                                                   dtype=torch.int)  # length + S + pad
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # Loss calc
            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut)

        # indexing
        unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features
    
    def cut_offset_anchor_features(self, offset, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            cut_xs = self.cut_xs + offset[batch_idx]
            cut_xs = torch.clamp(cut_xs, 0, self.fmap_w - 1).round().long()
            rois = img_features[self.cut_zs, self.cut_ys, cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def draw_anchors(self, img_w, img_h, k=None):
        base_ys = self.anchor_ys.numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for anchor in self.anchors:
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.numpy()
            xs = anchor[5:]
            ys = base_ys * img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=5)

        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.cut_zs = cuda_self.cut_zs.cuda(device)
        cuda_self.cut_ys = cuda_self.cut_ys.cuda(device)
        cuda_self.cut_xs = cuda_self.cut_xs.cuda(device)
        cuda_self.invalid_mask = cuda_self.invalid_mask.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.cut_zs = device_self.cut_zs.to(*args, **kwargs)
        device_self.cut_ys = device_self.cut_ys.to(*args, **kwargs)
        device_self.cut_xs = device_self.cut_xs.to(*args, **kwargs)
        device_self.invalid_mask = device_self.invalid_mask.to(*args, **kwargs)
        return device_self


def get_backbone(backbone, pretrained=False):
    if backbone == 'resnet122':
        backbone = resnet122_cifar()
        fmap_c = 64
        stride = 4
    elif backbone == 'resnet34':
        backbone = torch.nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    elif backbone == 'resnet18':
        backbone = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    else:
        raise NotImplementedError('Backbone not implemented: `{}`'.format(backbone))

    return backbone, fmap_c, stride
