# =============================================================================
# 🧪 Table 5: Ablation Study (Architecture & Pooling)
# 目的：证明 Late Fusion (你的方法) 优于 Early Fusion。
# 修复日志:
# 1. 修复 'Interaction' object has no attribute 'dataset' (传 d1 而非 d1[0])
# 2. 修复 TypeError: % int and NoneType (补全 n_heads, n_layers 等关键参数)
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shutil
import glob
import functools

# --- 0. 环境补丁 (Numpy 2.0 & PyTorch 2.6) ---
patch_map = {'float_': np.float64, 'int_': np.int64, 'bool_': bool, 'complex_': np.complex128, 'object_': object, 'unicode_': np.str_, 'string_': np.bytes_, 'str_': np.str_, 'float': float, 'int': int}
for alias, target in patch_map.items():
    if not hasattr(np, alias): setattr(np, alias, target)

_original_load = torch.load
torch.load = functools.partial(_original_load, weights_only=False)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.model.sequential_recommender import SASRec

# 🔥 统一导入：确保 src 目录下有 model_kgnn.py 🔥
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from model_kgnn import Gated_KG_SASRec
    print("✅ Successfully imported Gated_KG_SASRec from src.")
except ImportError:
    print("❌ Error: Could not import Gated_KG_SASRec. Please check src/model_kgnn.py")
    sys.exit(1)

# =============================================================================
# 🧩 变体模型定义 (Variants)
# =============================================================================

# 🔴 Variant 1: Early Fusion (前端融合)
class EarlyFusion_KGNN(SASRec):
    def __init__(self, config, dataset, kg_matrix, n_concepts):
        super(EarlyFusion_KGNN, self).__init__(config, dataset)
        self.neighbor_tensor = kg_matrix.to(self.device)
        self.concept_emb = nn.Embedding(n_concepts, config['hidden_size'], padding_idx=0)
        
    def forward(self, item_seq, item_seq_len):
        # 1. 基础 Item Embedding
        item_emb = self.item_embedding(item_seq)
        
        # 2. KG Lookup & Early Add
        neighbors = self.neighbor_tensor[item_seq] # [B, L, 10]
        neighbor_embs = self.concept_emb(neighbors) # [B, L, 10, H]
        # Mean Pooling
        mask = (neighbors != 0).unsqueeze(-1)
        neighbor_embs = neighbor_embs.masked_fill(~mask, 0.0)
        sum_emb = neighbor_embs.sum(dim=2)
        count = mask.sum(dim=2).clamp(min=1.0)
        kg_feat = sum_emb / count
        
        # 🔥 Early Fusion: 在进 Transformer 之前相加
        input_emb = item_emb + kg_feat
        
        # 3. Transformer Process
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        input_emb = self.LayerNorm(self.dropout(input_emb + position_embedding))
        trm_output = self.trm_encoder(input_emb, self.get_attention_mask(item_seq), output_all_encoded_layers=True)[-1]
        return self.gather_indexes(trm_output, item_seq_len - 1)

# --- 辅助函数 ---
def load_kg(dataset, data_dir):
    import pickle
    import re
    kg_path = os.path.join(data_dir, 'kg_data.pkl')
    if not os.path.exists(kg_path): kg_path = 'data/kg_data.pkl'
        
    with open(kg_path, 'rb') as f: raw_kg = pickle.load(f)
    target_data = raw_kg.get('item2concept', raw_kg)
    
    def extract_id(s):
        nums = re.findall(r'\d+', str(s))
        return int(nums[0]) if nums else None
    
    kg_map = {}
    if isinstance(target_data, dict):
        for k, v in target_data.items():
            d = extract_id(k)
            if d is not None: kg_map[d] = list(v)
            
    n_items = dataset.item_num
    matrix = np.zeros((n_items, 10), dtype=np.int32)
    item_id_list = dataset.field2id_token['item_id']
    for idx, token in enumerate(item_id_list):
        if idx == 0: continue 
        d = extract_id(token)
        if d in kg_map:
            cs = kg_map[d][:10]
            if len(cs) < 10: cs += [0]*(10-len(cs))
            matrix[idx] = cs
    return torch.LongTensor(matrix), 2000

def get_metrics_safe(result_dict):
    ndcg = result_dict.get('NDCG@10', result_dict.get('ndcg@10', 0.0))
    hit = result_dict.get('Hit@10', result_dict.get('hit@10', 0.0))
    return ndcg, hit

def clean_cache():
    root = os.getcwd()
    for d in ['dataset/MOOCCubeX', 'dataset/MOOCCubeX_KG_Only', 'saved']:
        path = os.path.join(root, d)
        if os.path.exists(path):
            if 'dataset' in d:
                for f in glob.glob(os.path.join(path, '*.pth')): os.remove(f)
            else:
                shutil.rmtree(path)

# --- 主实验逻辑 ---
def run_ablation():
    print("🧪 Running Ablation Study (Architecture Comparison)...")
    clean_cache()
    
    config_dict = {
        'data_path': 'dataset/',
        'dataset': 'MOOCCubeX',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'loss_type': 'BPR',
        'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1},
        'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'uni100'},
        'metrics': ['NDCG', 'Hit'], 'topk': [10], 'valid_metric': 'ndcg@10',
        'seed': 2024, 'gpu_id': 0, 'use_gpu': True,
        'epochs': 20, 'stopping_step': 5,
        'MAX_ITEM_LIST_LENGTH': 5,
        'hidden_size': 64, 'inner_size': 256, 'embedding_size': 64,
        'state': 'WARNING',
        
        # 🔥 [关键修复] 补全 SASRec 必须的架构参数 🔥
        'n_layers': 2, 
        'n_heads': 2, 
        'hidden_act': 'gelu', 
        'dropout_prob': 0.2, 
        'attn_dropout_prob': 0.2, 
        'hidden_dropout_prob': 0.2,
        'initializer_range': 0.02, 
        'layer_norm_eps': 1e-12
    }
    
    results = {}

    # --- 1. Run Early Fusion ---
    print("\n🔹 [Variant 1] Running Early Fusion...")
    c1 = Config(model=EarlyFusion_KGNN, dataset='MOOCCubeX', config_dict=config_dict)
    init_seed(c1['seed'], c1['reproducibility'])
    d1 = create_dataset(c1)
    d1_prep = data_preparation(c1, d1)
    
    data_dir = 'data' if os.path.exists('data/kg_data.pkl') else os.path.join(os.getcwd(), 'data')
    kg_mat, n_con = load_kg(d1, data_dir)
    
    # 修复调用方式
    m1 = EarlyFusion_KGNN(c1, d1, kg_mat, n_con).to(c1['device'])
    t1 = Trainer(c1, m1)
    t1.fit(d1_prep[0], d1_prep[1], verbose=False)
    res1 = t1.evaluate(d1_prep[2])
    
    ndcg1, hit1 = get_metrics_safe(res1)
    results['Early Fusion'] = {'NDCG': ndcg1, 'Hit': hit1}
    print(f"   >>> Early Fusion: NDCG@10={ndcg1:.4f} | Hit@10={hit1:.4f}")
    
    # --- 2. Run Late Fusion (Ours) ---
    print("\n🔹 [Ours] Running Late Fusion (K-GNN)...")
    clean_cache() 
    c2 = Config(model=Gated_KG_SASRec, dataset='MOOCCubeX', config_dict=config_dict)
    init_seed(c2['seed'], c2['reproducibility'])
    d2 = create_dataset(c2)
    d2_prep = data_preparation(c2, d2)
    
    m2 = Gated_KG_SASRec(c2, d2, kg_mat, n_con).to(c2['device'])
    t2 = Trainer(c2, m2)
    t2.fit(d2_prep[0], d2_prep[1], verbose=False)
    res2 = t2.evaluate(d2_prep[2])
    
    ndcg2, hit2 = get_metrics_safe(res2)
    results['Late Fusion (Ours)'] = {'NDCG': ndcg2, 'Hit': hit2}
    print(f"   >>> Late Fusion:  NDCG@10={ndcg2:.4f} | Hit@10={hit2:.4f}")

    # --- Final Report ---
    print("\n" + "="*50)
    print("🧪 ABLATION STUDY RESULTS (Table 5)")
    print("="*50)
    print(f"{'Variant':<20} | {'NDCG@10':<10} | {'Hit@10':<10}")
    print("-" * 50)
    
    r1 = results['Early Fusion']
    print(f"{'Early Fusion':<20} | {r1['NDCG']:.4f}     | {r1['Hit']:.4f}")
    
    r2 = results['Late Fusion (Ours)']
    print(f"{'Late Fusion (Ours)':<20} | {r2['NDCG']:.4f}     | {r2['Hit']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    run_ablation()