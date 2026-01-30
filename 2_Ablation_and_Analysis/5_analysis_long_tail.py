# =============================================================================
# 📊 Tail Analysis: Diagnosing the "Where do we win?" (Final Fix)
# 修复点:
# 1. Tensor .cpu().numpy() 转换 (解决 value_counts 报错)
# 2. loss_type='BPR' (解决负采样配置冲突)
# 3. Numpy 2.0 兼容补丁
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd

# 🔥 0. Numpy 2.0 补丁 (必须在最前面)
patch_map = {
    'float_': np.float64, 'int_': np.int64, 'bool_': bool,
    'complex_': np.complex128, 'object_': object,
    'unicode_': np.str_, 'string_': np.bytes_, 'str_': np.str_,
    'float': float, 'int': int
}
for alias, target in patch_map.items():
    if not hasattr(np, alias): setattr(np, alias, target)

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from recbole.model.sequential_recommender import SASRec

def analyze_performance_by_popularity():
    print("🕵️‍♂️ 正在执行【长尾性能分析】...")
    
    # 1. 重新加载配置
    current_path = os.getcwd()
    dataset_path = os.path.join(current_path, 'dataset')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(current_path, 'K-GNN_Replication_Package', 'dataset')
    
    print(f"   📂 数据集路径: {dataset_path}")

    config_dict = {
        'data_path': dataset_path,
        'dataset': 'MOOCCubeX',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        # ✅ 修复 1: 显式指定 loss_type='BPR'
        'loss_type': 'BPR',
        'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1},
        'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'uni100'},
        'metrics': ['NDCG', 'Hit'], 'topk': [10],
        'seed': 2024, 'gpu_id': 0, 'use_gpu': True,
        'MAX_ITEM_LIST_LENGTH': 5,
        'state': 'INFO'
    }
    
    try:
        config = Config(model=SASRec, dataset='MOOCCubeX', config_dict=config_dict)
        init_seed(config['seed'], config['reproducibility'])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 计算物品流行度 (Popularity)
    print("   📊 计算物品流行度分布...")
    
    # ✅ 修复 2: Tensor -> Numpy 转换
    # RecBole 的 inter_feat['item_id'] 是个 Tensor，不能直接 value_counts
    item_id_tensor = dataset.inter_feat['item_id']
    if torch.is_tensor(item_id_tensor):
        item_id_numpy = item_id_tensor.cpu().numpy()
    else:
        item_id_numpy = item_id_tensor
        
    item_freq = pd.Series(item_id_numpy).value_counts().sort_index()
    
    # 补全那些没出现的 item 为 0
    all_items = np.arange(dataset.item_num)
    freq_map = pd.Series(0, index=all_items)
    freq_map.update(item_freq)
    
    # 除去 padding (0)
    if 0 in freq_map.index: freq_map = freq_map.drop(0)
    
    print(f"   统计概览:\n{freq_map.describe()}")
    
    # 定义分组 (Hot/Cold)
    def get_group(count):
        if count <= 5: return "1_Very_Cold (<5)"
        elif count <= 10: return "2_Cold (5-10)"
        elif count <= 20: return "3_Mid (10-20)"
        elif count <= 50: return "4_Warm (20-50)"
        else: return "5_Hot (>50)"
        
    group_map = freq_map.apply(get_group)
    
    # 3. 分析测试集分布
    print("\n   🎯 分析测试集 (Test Set) 的目标物品分布...")
    target_items = []
    
    # 遍历 DataLoader
    for batch_idx, batched_data in enumerate(test_data):
        # 兼容处理：有些版本返回 tuple，有些直接返回 Interaction 对象
        interaction = batched_data[0] if isinstance(batched_data, (tuple, list)) else batched_data
        
        if 'item_id' in interaction:
            target_items.extend(interaction['item_id'].cpu().numpy())
            
    if not target_items:
        print("⚠️ 无法从 Loader 提取，跳过详细分布分析。")
        return

    target_groups = [group_map.get(i, "Unknown") for i in target_items]
    target_group_counts = pd.Series(target_groups).value_counts().sort_index()
    
    print("\n   🧪 测试集物品热度分布 (Model Exam Questions):")
    total = len(target_items)
    for g in target_group_counts.index:
        count = target_group_counts[g]
        pct = (count / total) * 100
        print(f"   {g}: {count} ({pct:.2f}%)")
    
    # 4. 诊断结论
    hot_ratio = (target_group_counts.get("5_Hot (>50)", 0) + target_group_counts.get("4_Warm (20-50)", 0)) / total
    print("\n   💡 核心诊断:")
    print(f"   热门物品 (Warm+Hot) 占比: {hot_ratio*100:.2f}%")
    
    if hot_ratio > 0.5:
        print("   🔴 结论: 测试集被【热门物品】主导了！")
        print("   这就像是在考‘长尾分布’，但卷子里 80% 的题都是送分题。")
        print("   SASRec 拿高分是因为它只做了送分题。")
        print("   K-GNN 的价值在于那 20% 的难题，但被平均分淹没了。")
    else:
        print("   🟢 结论: 测试集分布相对均衡，K-GNN 理应在总分上有体现。")

if __name__ == "__main__":
    analyze_performance_by_popularity()