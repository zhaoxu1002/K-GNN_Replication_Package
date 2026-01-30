# =============================================================================
# 🏆 Table 4: Robustness Analysis (Final Refactored Version)
# - Models: SASRec, BERT4Rec, GRU4Rec, K-GNN (Imported from src)
# - Metrics: NDCG, Hit, Recall, Precision
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

# --- 1. Numpy 2.0 Patch ---
patch_map = {'float_': np.float64, 'int_': np.int64, 'bool_': bool, 'complex_': np.complex128, 'object_': object, 'unicode_': np.str_, 'string_': np.bytes_, 'str_': np.str_, 'float': float, 'int': int}
for alias, target in patch_map.items():
    if not hasattr(np, alias): setattr(np, alias, target)

# --- 2. PyTorch 2.6 Patch ---
_original_load = torch.load
torch.load = functools.partial(_original_load, weights_only=False)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.model.sequential_recommender import SASRec, BERT4Rec, GRU4Rec

# 🔥 Refactor: Import from src instead of inline definition 🔥
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from model_kgnn import Gated_KG_SASRec
    print("✅ Successfully imported Gated_KG_SASRec from src.")
except ImportError:
    print("❌ Error: Could not import Gated_KG_SASRec. Please ensure 'src/model_kgnn.py' exists.")
    sys.exit(1)

# --- 辅助函数 ---
def load_kg(dataset, data_dir):
    import pickle
    import re
    kg_path = os.path.join(data_dir, 'kg_data.pkl')
    if not os.path.exists(kg_path): return torch.zeros((dataset.item_num, 10), dtype=torch.long), 1
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
    all_concepts = set()
    for idx, token in enumerate(item_id_list):
        if idx == 0: continue 
        d = extract_id(token)
        if d in kg_map:
            cs = kg_map[d][:10]
            if len(cs) < 10: cs += [0]*(10-len(cs))
            matrix[idx] = cs
            all_concepts.update(cs)
    return torch.LongTensor(matrix), (max(all_concepts) + 1 if all_concepts else 2000)

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
    # 尝试匹配 RecBole 可能返回的不同 Key 格式 (例如 'NDCG@10', 'ndcg@10')
    candidates = [metric_name, metric_name.lower(), metric_name.upper()]
    for key in candidates:
        if key in result_dict: return result_dict[key]
    return 0.0

# --- 主实验逻辑 ---
def run_sparsity_experiment():
    print("🚀 启动全能鲁棒性实验 (SASRec, BERT4Rec, GRU4Rec, K-GNN)...")
    
    # Ratios
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Init Results Dictionary
    # We will track NDCG@10 primarily for the plot, but print others
    results = {'Ratio': ratios, 'SASRec': [], 'BERT4Rec': [], 'GRU4Rec': [], 'K-GNN': []}
    
    original_inter = 'dataset/MOOCCubeX/MOOCCubeX.inter'
    if not os.path.exists(original_inter):
        # 兼容路径
        original_inter = 'K-GNN_Replication_Package/dataset/MOOCCubeX/MOOCCubeX.inter'
    
    if not os.path.exists(original_inter):
        print(f"❌ 找不到数据文件: {original_inter}")
        return

    backup_inter = original_inter + '.bak'
    shutil.copy(original_inter, backup_inter)
    
    try:
        df_full = pd.read_csv(backup_inter, sep='\t')
        
        for ratio in ratios:
            print(f"\n⚡ Data Ratio: {ratio*100}%")
            
            # Sampling
            if ratio < 1.0:
                df_sampled = df_full.sample(frac=ratio, random_state=2024)
            else:
                df_sampled = df_full
            
            df_sampled.to_csv(original_inter, sep='\t', index=False)
            
            clean_cache()
            
            # --- 通用 BPR 配置 ---
            base_config = {
                'data_path': os.path.dirname(os.path.dirname(original_inter)),
                'dataset': 'MOOCCubeX',
                'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
                'loss_type': 'BPR',
                'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1},
                'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'uni100'},
                
                # 🔥 Added Metrics: Recall, Precision 🔥
                'metrics': ['NDCG', 'Hit', 'Recall', 'Precision'], 
                'topk': [10],
                'valid_metric': 'ndcg@10', 
                
                'seed': 2024, 'gpu_id': 0, 'use_gpu': True, 'worker': 0,
                'epochs': 20, 'stopping_step': 3,
                'MAX_ITEM_LIST_LENGTH': 5, # 统一配置
                
                'state': 'WARNING', 
                'learning_rate': 0.001,
                'hidden_size': 64, 'inner_size': 256, 'embedding_size': 64, # 统一配置
                'n_layers': 2, 'n_heads': 2, 
                'hidden_act': 'gelu', 'dropout_prob': 0.2, 'attn_dropout_prob': 0.2, 'hidden_dropout_prob': 0.2,
                'initializer_range': 0.02, 'layer_norm_eps': 1e-12
            }
            
            # --- 1. SASRec ---
            print("   Running SASRec...")
            conf_s = Config(model=SASRec, dataset='MOOCCubeX', config_dict=base_config)
            init_seed(conf_s['seed'], conf_s['reproducibility'])
            data_s = data_preparation(conf_s, create_dataset(conf_s))
            model_s = SASRec(conf_s, data_s[0].dataset).to(conf_s['device'])
            trainer_s = Trainer(conf_s, model_s)
            trainer_s.fit(data_s[0], data_s[1], verbose=False)
            res_s = trainer_s.evaluate(data_s[2])
            results['SASRec'].append(get_metric(res_s, 'NDCG@10'))
            # 🔥 显式打印 🔥
            print(f"   >>> SASRec:   NDCG: {results['SASRec'][-1]:.4f} | Hit: {get_metric(res_s, 'Hit@10'):.4f} | Recall: {get_metric(res_s, 'Recall@10'):.4f} | Prec: {get_metric(res_s, 'Precision@10'):.4f}")

            # --- 2. GRU4Rec ---
            print("   Running GRU4Rec...")
            conf_g = Config(model=GRU4Rec, dataset='MOOCCubeX', config_dict=base_config)
            init_seed(conf_g['seed'], conf_g['reproducibility'])
            data_g = data_preparation(conf_g, create_dataset(conf_g))
            model_g = GRU4Rec(conf_g, data_g[0].dataset).to(conf_g['device'])
            trainer_g = Trainer(conf_g, model_g)
            trainer_g.fit(data_g[0], data_g[1], verbose=False)
            res_g = trainer_g.evaluate(data_g[2])
            results['GRU4Rec'].append(get_metric(res_g, 'NDCG@10'))
            print(f"   >>> GRU4Rec:  NDCG: {results['GRU4Rec'][-1]:.4f} | Hit: {get_metric(res_g, 'Hit@10'):.4f} | Recall: {get_metric(res_g, 'Recall@10'):.4f} | Prec: {get_metric(res_g, 'Precision@10'):.4f}")

            # --- 3. BERT4Rec (CE Loss) ---
            print("   Running BERT4Rec...")
            bert_config = base_config.copy()
            bert_config['loss_type'] = 'CE' 
            bert_config['train_neg_sample_args'] = None 
            bert_config['mask_ratio'] = 0.2 
            
            conf_b = Config(model=BERT4Rec, dataset='MOOCCubeX', config_dict=bert_config)
            init_seed(conf_b['seed'], conf_b['reproducibility'])
            data_b = data_preparation(conf_b, create_dataset(conf_b))
            model_b = BERT4Rec(conf_b, data_b[0].dataset).to(conf_b['device'])
            trainer_b = Trainer(conf_b, model_b)
            trainer_b.fit(data_b[0], data_b[1], verbose=False)
            res_b = trainer_b.evaluate(data_b[2])
            results['BERT4Rec'].append(get_metric(res_b, 'NDCG@10'))
            print(f"   >>> BERT4Rec: NDCG: {results['BERT4Rec'][-1]:.4f} | Hit: {get_metric(res_b, 'Hit@10'):.4f} | Recall: {get_metric(res_b, 'Recall@10'):.4f} | Prec: {get_metric(res_b, 'Precision@10'):.4f}")

            # --- 4. K-GNN (Ours) ---
            print("   Running K-GNN...")
            conf_k = Config(model=Gated_KG_SASRec, dataset='MOOCCubeX', config_dict=base_config)
            init_seed(conf_k['seed'], conf_k['reproducibility'])
            data_k = data_preparation(conf_k, create_dataset(conf_k))
            
            data_dir = os.path.dirname(original_inter).replace('dataset/MOOCCubeX', 'data')
            if not os.path.exists(data_dir): data_dir = 'data'
            kg_matrix, n_concepts = load_kg(data_k[0].dataset, data_dir)
            
            model_k = Gated_KG_SASRec(conf_k, data_k[0].dataset, kg_matrix, n_concepts).to(conf_k['device'])
            trainer_k = Trainer(conf_k, model_k)
            trainer_k.fit(data_k[0], data_k[1], verbose=False)
            res_k = trainer_k.evaluate(data_k[2])
            results['K-GNN'].append(get_metric(res_k, 'NDCG@10'))
            print(f"   >>> K-GNN:    NDCG: {results['K-GNN'][-1]:.4f} | Hit: {get_metric(res_k, 'Hit@10'):.4f} | Recall: {get_metric(res_k, 'Recall@10'):.4f} | Prec: {get_metric(res_k, 'Precision@10'):.4f}")

    finally:
        if os.path.exists(backup_inter):
            shutil.move(backup_inter, original_inter)
        print("\n✅ 实验结束，数据已还原。")

    # --- 绘图 ---
    print("\n📊 绘制全能鲁棒性曲线...")
    df_res = pd.DataFrame(results)
    print(df_res)
    
    plt.figure(figsize=(9, 6))
    plt.plot(df_res['Ratio'], df_res['SASRec'], marker='o', label='SASRec', linestyle='--')
    plt.plot(df_res['Ratio'], df_res['GRU4Rec'], marker='^', label='GRU4Rec', linestyle='--')
    plt.plot(df_res['Ratio'], df_res['BERT4Rec'], marker='x', label='BERT4Rec', linestyle='--')
    plt.plot(df_res['Ratio'], df_res['K-GNN'], marker='s', label='K-GNN (Ours)', linewidth=2.5, color='red')
    
    plt.xlabel('Training Data Ratio')
    plt.ylabel('NDCG@10')
    plt.title('Robustness Analysis: K-GNN vs SOTA Baselines')
    plt.legend()
    plt.grid(True)
    plt.savefig('robustness_all_baselines.png')
    print("📷 图表已保存: robustness_all_baselines.png")

if __name__ == "__main__":
    run_sparsity_experiment()