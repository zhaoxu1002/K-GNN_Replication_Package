# =============================================================================
# 🏆 Table 3: Extreme Cold-Start (Late Fusion / Post-Transformer Injection)
# 核心策略:
# 1. 保护 SASRec: 输入端不加任何 KG，确保 Transformer 性能不受损。
# 2. 后端注入 (Late Fusion): 在 Transformer 输出层注入 KG 信息。
# 3. 逻辑: "先按序列猜 (SASRec)，再用知识微调 (KG)"。
# =============================================================================

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
import functools
import time
import glob
import shutil

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

# 🔥 Refactor: Import from src instead of inline definition 🔥
# 确保 src 文件夹在当前目录下，或者在 PYTHONPATH 中
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from model_kgnn import Gated_KG_SASRec
    print("✅ Successfully imported Gated_KG_SASRec from src.")
except ImportError:
    print("❌ Error: Could not import Gated_KG_SASRec. Please ensure 'src/model_kgnn.py' exists.")
    # Fallback: 如果实在找不到，提示用户
    sys.exit(1)

# --- 1. 辅助函数 ---
def load_and_check_kg(dataset):
    print("📦 Loading KG Matrix...")
    # 尝试定位 data 目录
    root_dir = os.path.dirname(os.path.dirname(dataset.config['data_path']))
    kg_path = os.path.join(root_dir, 'data', 'kg_data.pkl')
    
    if not os.path.exists(kg_path):
        # 备用路径 (Kaggle/Colab 常见路径)
        kg_path = 'data/kg_data.pkl'
        if not os.path.exists(kg_path):
            print(f"❌ KG File not found at {kg_path}")
            sys.exit(1)

    with open(kg_path, 'rb') as f:
        raw_kg = pickle.load(f)
    
    # 兼容处理:有些 pkl 直接是 dict，有些是 {'item2concept': ...}
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
    
    all_concepts = set()
    hit_count = 0
    
    for idx, token in enumerate(item_id_list):
        if idx == 0: continue 
        d = extract_id(token)
        if d in kg_map:
            cs = kg_map[d][:10]
            if len(cs) < 10: cs += [0]*(10-len(cs))
            matrix[idx] = cs
            all_concepts.update(cs)
            hit_count += 1
            
    print(f"   📊 KG Coverage: {hit_count}/{n_items} items ({hit_count/n_items:.2%})")
    n_concepts = (max(all_concepts) + 1) if all_concepts else 2000
    return torch.LongTensor(matrix), n_concepts

def clean_cache():
    root = os.getcwd()
    for d in ['dataset/MOOCCubeX', 'dataset/MOOCCubeX_KG_Only', 'saved']:
        path = os.path.join(root, d)
        if os.path.exists(path):
            if 'dataset' in d:
                for f in glob.glob(os.path.join(path, '*.pth')): os.remove(f)
            else:
                shutil.rmtree(path)

# --- 2. 主实验逻辑 ---
def run_main_experiment():
    print("🚀 启动主实验 (Table 3 Comparison)...")
    clean_cache()
    
    DATASET_NAME = 'MOOCCubeX'
    
    # 基础配置 (SASRec & K-GNN 共享)
    base_config = {
        'data_path': 'dataset/',
        'dataset': DATASET_NAME,
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'loss_type': 'BPR',
        'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1},
        'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'uni100'},
        
        # 🔥 Metrics Updated: Added Recall, Precision 🔥
        'metrics': ['NDCG', 'Hit', 'Recall', 'Precision'],
        'topk': [10],
        'valid_metric': 'ndcg@10',
        
        'seed': 2024,
        'gpu_id': 0, 'use_gpu': True, 'worker': 0,
        'epochs': 20, 'stopping_step': 5,
        'MAX_ITEM_LIST_LENGTH': 5, # 统一配置
        'learning_rate': 0.001,
        'hidden_size': 64, 'inner_size': 256, 'embedding_size': 64, # 统一配置
        'n_layers': 2, 'n_heads': 2, 
        'hidden_act': 'gelu', 'dropout_prob': 0.2, 'attn_dropout_prob': 0.2, 'hidden_dropout_prob': 0.2,
        'initializer_range': 0.02, 'layer_norm_eps': 1e-12,
        'state': 'INFO' # Show training loss logs
    }

    # A. SASRec
    print("\n🔹 [1/2] Running Baseline: SASRec ...")
    conf_s = Config(model=SASRec, dataset=DATASET_NAME, config_dict=base_config)
    init_seed(conf_s['seed'], conf_s['reproducibility'])
    dataset_s = create_dataset(conf_s)
    train_s, valid_s, test_s = data_preparation(conf_s, dataset_s)
    
    model_s = SASRec(conf_s, train_s.dataset).to(conf_s['device'])
    trainer_s = Trainer(conf_s, model_s)
    trainer_s.fit(train_s, valid_s, verbose=True) 
    res_s = trainer_s.evaluate(test_s)
    
    # B. K-GNN
    print("\n🔹 [2/2] Running Ours: K-GNN (Late Fusion) ...")
    conf_k = Config(model=Gated_KG_SASRec, dataset=DATASET_NAME, config_dict=base_config)
    init_seed(conf_k['seed'], conf_k['reproducibility'])
    dataset_k = create_dataset(conf_k)
    
    kg_matrix, n_concepts = load_and_check_kg(dataset_k)
    
    train_k, valid_k, test_k = data_preparation(conf_k, dataset_k)
    model_k = Gated_KG_SASRec(conf_k, train_k.dataset, kg_matrix, n_concepts).to(conf_k['device'])
    trainer_k = Trainer(conf_k, model_k)
    trainer_k.fit(train_k, valid_k, verbose=True)
    res_k = trainer_k.evaluate(test_k)

    # C. 结果报告 (修复版：包含 Precision)
    print("\n" + "="*80)
    print("🏆 FINAL RESULT REPORT (Full Metrics)")
    print("="*80)
    # 调整列宽以容纳所有指标
    print(f"{'Model':<10} | {'NDCG@10':<10} | {'Hit@10':<10} | {'Recall@10':<10} | {'Prec@10':<10}")
    print("-" * 80)
    
    def safe_get(d, k): 
        # 兼容 RecBole 的大小写 (ndcg@10 或 NDCG@10)
        return d.get(k, d.get(k.lower(), 0.0))
    
    # 获取指标
    sas_ndcg = safe_get(res_s, 'NDCG@10')
    sas_hit = safe_get(res_s, 'Hit@10')
    sas_recall = safe_get(res_s, 'Recall@10')
    sas_prec = safe_get(res_s, 'Precision@10')
    
    kgnn_ndcg = safe_get(res_k, 'NDCG@10')
    kgnn_hit = safe_get(res_k, 'Hit@10')
    kgnn_recall = safe_get(res_k, 'Recall@10')
    kgnn_prec = safe_get(res_k, 'Precision@10')

    print(f"{'SASRec':<10} | {sas_ndcg:.4f}     | {sas_hit:.4f}     | {sas_recall:.4f}     | {sas_prec:.4f}")
    print(f"{'K-GNN':<10} | {kgnn_ndcg:.4f}     | {kgnn_hit:.4f}     | {kgnn_recall:.4f}     | {kgnn_prec:.4f}")
    print("-" * 80)
    
    # 验证数学关系 (可选)
    # print(f"Check: SASRec Hit({sas_hit}) == Recall({sas_recall})? {abs(sas_hit-sas_recall) < 1e-5}")

if __name__ == "__main__":
    run_main_experiment()