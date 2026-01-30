# -*- coding: utf-8 -*-
# @Time   : 2026/01/21
# @Author : K-GNN Replication Team
# @Email  : replication@research.com

r"""
K-GNN (Knowledge-Enhanced Sequential Recommender)
################################################
Reference:
    Late Fusion / Post-Transformer Injection Architecture
    (Proven to be the best variant for Mixed Data Cold-Start)
"""

import torch
import torch.nn as nn
from recbole.model.sequential_recommender import SASRec
from recbole.utils import InputType

class Gated_KG_SASRec(SASRec):
    r"""
    K-GNN with Late Fusion Architecture.
    
    Architecture:
    1. Standard SASRec Encoder (Item Seq -> Transformer -> Hidden State)
    2. KG Lookup (Last Item -> Concepts -> Max Pooling)
    3. Late Fusion (Hidden State + KG Embedding)
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, kg_matrix, n_concepts):
        super(Gated_KG_SASRec, self).__init__(config, dataset)

        # 加载 KG 邻居矩阵 [Item_Num, 10]
        self.neighbor_tensor = kg_matrix.to(self.device)
        
        # 概念 Embedding (padding_idx=0 保证空邻居不产生噪音)
        # config['hidden_size'] 必须在外部保证存在
        self.concept_emb = nn.Embedding(n_concepts, config['hidden_size'], padding_idx=0)

        # --- 后端融合层 (Late Fusion) ---
        # 将 KG 特征映射到与序列特征相同的空间，并做非线性变换
        self.post_fusion = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.Tanh() # Tanh 限制幅度，防止 KG 噪音过大
        )
        
        # 最终的 LayerNorm，保证融合后的分布稳定
        self.fusion_norm = nn.LayerNorm(config['hidden_size'])

    def forward(self, item_seq, item_seq_len):
        # 1. --- 标准 SASRec 流程 (Standard Sequence Encoding) ---
        item_emb = self.item_embedding(item_seq)
        
        # 位置编码
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        # Input Embedding
        input_emb = self.LayerNorm(self.dropout(item_emb + position_embedding))
        
        # Transformer Encoding
        # output_all_encoded_layers=True returns a list, we take the last layer [-1]
        trm_output = self.trm_encoder(input_emb, self.get_attention_mask(item_seq), output_all_encoded_layers=True)[-1]
        
        # Gather the hidden state at the last time step (Target User Intent)
        seq_output = self.gather_indexes(trm_output, item_seq_len - 1)
        
        # 2. --- Late Fusion (KG Injection) ---
        # 获取序列中最后一个 Item 的 ID (User 当前感兴趣的物品)
        last_item_indices = item_seq_len - 1
        # gather: [B, L] -> [B, 1] -> [B]
        last_items = item_seq.gather(1, last_item_indices.unsqueeze(1)).squeeze(1)
        
        # KG Lookup: [B] -> [B, 10]
        neighbors = self.neighbor_tensor[last_items] 
        # Concept Embedding: [B, 10, H]
        neighbor_embs = self.concept_emb(neighbors) 
        
        # Max Pooling (Extract the most salient concept)
        mask = (neighbors != 0).unsqueeze(-1) # [B, 10, 1]
        # Mask out padding with a very small number
        neighbor_embs = neighbor_embs.masked_fill(~mask, -1e9)
        # Max over dim=1 (neighbors) -> [B, H]
        kg_feat, _ = torch.max(neighbor_embs, dim=1)
        
        # Handle case where all neighbors are 0 (no KG info) -> result is -1e9 -> reset to 0
        kg_feat = kg_feat.masked_fill(kg_feat < -1e8, 0.0)
        
        # 3. --- Fusion & Output ---
        # Sequence State + KG Bias
        kg_bias = self.post_fusion(kg_feat)
        final_output = seq_output + kg_bias
        
        return self.fusion_norm(final_output)

    def calculate_loss(self, interaction):
        return super(Gated_KG_SASRec, self).calculate_loss(interaction)