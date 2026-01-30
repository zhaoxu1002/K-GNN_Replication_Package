# =============================================================================
# 📊 Figure 5: Parameter Sensitivity Analysis
# 目的：探究 hidden_size (32, 64, 128) 对模型性能的影响
# 修复：使用 src 导入模型，保证代码统一性
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shutil
import glob
import functools

# --- 0. 环境补丁 ---
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

# 🔥 Refactor: Import from src 🔥
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from model_kgnn import Gated_KG_SASRec
except ImportError:
    print("❌ Error: Could not import Gated_KG_SASRec. Check src/model_kgnn.py")
    sys.exit(1)

# --- 辅助函数 ---
def load_kg(dataset, data_dir):
    import pickle
    import re
    kg_path = os.path.join(data_dir, 'kg_data.pkl')
    if not os.path.exists(kg_path): 
        # Fallback
        kg_path = 'data/kg_data.pkl'
        
    if not os.path.exists(kg_path):
        return torch.zeros((dataset.item_num, 10), dtype=torch.long), 1
        
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

def clean_cache():
    root = os.getcwd()
    for d in ['dataset/MOOCCubeX', 'dataset/MOOCCubeX_KG_Only', 'saved']:
        path = os.path.join(root, d)
        if os.path.exists(path):
            if 'dataset' in d:
                for f in glob.glob(os.path.join(path, '*.pth')): os.remove(f)
            else:
                shutil.rmtree(path)

def get_metric(result_dict, metric_name):
    candidates = [metric_name, metric_name.lower(), metric_name.upper()]
    for key in candidates:
        if key in result_dict: return result_dict[key]
    return 0.0

# --- 主逻辑 ---
def run_sensitivity():
    print("📊 Parameter Sensitivity Analysis (Hidden Size)...")
    
    original_inter = 'dataset/MOOCCubeX/MOOCCubeX.inter'
    if not os.path.exists(original_inter):
        original_inter = 'K-GNN_Replication_Package/dataset/MOOCCubeX/MOOCCubeX.inter'
    
    # 动态调整 hidden_size，其他保持统一
    hidden_sizes = [32, 64, 128]
    results = {'Hidden Size': hidden_sizes, 'NDCG@10': []}

    for d_val in hidden_sizes:
        print(f"\n⚡ Testing Hidden Size: {d_val}")
        clean_cache()
        
        config_dict = {
            'data_path': os.path.dirname(os.path.dirname(original_inter)),
            'dataset': 'MOOCCubeX',
            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
            'loss_type': 'BPR',
            'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1},
            'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'uni100'},
            'metrics': ['NDCG', 'Hit'], 'topk': [10], 'valid_metric': 'ndcg@10',
            'seed': 2024, 'gpu_id': 0, 'use_gpu': True,
            'epochs': 15, 'stopping_step': 3,
            'MAX_ITEM_LIST_LENGTH': 5, # 统一配置
            'state': 'WARNING',
            
            # 🔥 动态调整参数 🔥
            'hidden_size': d_val,
            'inner_size': d_val * 4,
            'embedding_size': d_val,
            
            # 固定参数 (与 3_exp 保持一致)
            'learning_rate': 0.001,
            'n_layers': 2, 'n_heads': 2, 
            'hidden_act': 'gelu', 
            'dropout_prob': 0.2, 'attn_dropout_prob': 0.2, 'hidden_dropout_prob': 0.2,
            'initializer_range': 0.02, 'layer_norm_eps': 1e-12
        }
        
        # Run K-GNN
        c = Config(model=Gated_KG_SASRec, dataset='MOOCCubeX', config_dict=config_dict)
        init_seed(c['seed'], c['reproducibility'])
        d = data_preparation(c, create_dataset(c))
        
        data_dir = os.path.dirname(original_inter).replace('dataset/MOOCCubeX', 'data')
        if not os.path.exists(data_dir): data_dir = 'data'
        kg_mat, n_con = load_kg(d[0].dataset, data_dir)
        
        m = Gated_KG_SASRec(c, d[0].dataset, kg_mat, n_con).to(c['device'])
        t = Trainer(c, m)
        t.fit(d[0], d[1], verbose=False)
        res = t.evaluate(d[2])
        score = get_metric(res, 'NDCG@10')
        results['NDCG@10'].append(score)
        print(f"   >>> d={d_val} | NDCG@10: {score:.4f}")

    # --- 绘图 ---
    print("\n📈 绘制敏感性分析图...")
    plt.figure(figsize=(6, 4))
    plt.plot(results['Hidden Size'], results['NDCG@10'], marker='o', linewidth=2, color='purple')
    plt.xlabel('Hidden Size (d)')
    plt.ylabel('NDCG@10')
    plt.title('Parameter Sensitivity Analysis')
    plt.xticks(hidden_sizes)
    plt.grid(True)
    plt.savefig('parameter_sensitivity.png')
    print("📷 图表已保存: parameter_sensitivity.png")
    
    print("\n✅ 结果数据:")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    run_sensitivity()