# =============================================================================
# ⏱️ Script 9: Measure Inference Latency (Fixed Warmup)
# 修复: 将预热次数从 10 降低到 2，防止跳过所有测试数据
# =============================================================================

import sys
import os
import time
import numpy as np
import torch
import functools

# --- 环境补丁 ---
patch_map = {'float_': np.float64, 'int_': np.int64, 'bool_': bool, 'complex_': np.complex128, 'object_': object, 'unicode_': np.str_, 'string_': np.bytes_, 'str_': np.str_, 'float': float, 'int': int}
for alias, target in patch_map.items():
    if not hasattr(np, alias): setattr(np, alias, target)

_original_load = torch.load
torch.load = functools.partial(_original_load, weights_only=False)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec

sys.path.append(os.path.join(os.getcwd(), 'src')) 
try:
    from model_kgnn import Gated_KG_SASRec
except:
    pass

def measure_time(model, test_data, device, model_name="Model"):
    model.eval()
    latencies = []
    
    # 获取测试集总 Batch 数
    total_batches = len(test_data)
    # 动态调整预热次数：如果数据很少，就只预热 1 次，否则预热 2 次
    warmup_steps = 1 if total_batches < 5 else 2
    
    print(f"\n🚀 开始测试 {model_name} (Total Batches: {total_batches})...")
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_data):
            interaction = batch_data[0].to(device)
            
            # 1. 预热
            if i < warmup_steps:
                _ = model.full_sort_predict(interaction)
                continue
            
            # 2. 精确计时
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = model.full_sort_predict(interaction)
            end_event.record()
            
            torch.cuda.synchronize()
            
            elapsed_time_ms = start_event.elapsed_time(end_event)
            
            # 获取当前 Batch 的用户数
            try:
                batch_users = interaction.user_id.size(0)
            except:
                batch_users = interaction[0].size(0)

            latencies.append(elapsed_time_ms / batch_users)
            
            # 测 50 个 Batch 就够稳定了
            if i >= (warmup_steps + 50): break
    
    if len(latencies) == 0:
        print("❌ 错误: 没有收集到耗时数据，请检查测试集是否为空！")
        return 0.0

    avg_latency = np.mean(latencies)
    throughput = 1000 / avg_latency if avg_latency > 0 else 0
    print(f"   ✅ 平均推理延迟 (Latency): {avg_latency:.4f} ms/user")
    print(f"   ✅ 吞吐量 (Throughput):    {throughput:.2f} users/sec")
    return avg_latency

if __name__ == '__main__':
    # 1. 准备配置
    config_dict = {
        'use_gpu': True, 'gpu_id': 0, 'state': 'INFO',
        'eval_args': {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'},
        'topk': [10],
        'loss_type': 'BPR',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']}, 
        'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0}
    }
    
    config = Config(model='SASRec', dataset='MOOCCubeX', config_dict=config_dict)
    dataset = create_dataset(config)
    _, _, test_data = data_preparation(config, dataset)
    
    # 2. 初始化模型用于测速 (随机权重即可，不影响计算图结构和速度)
    print("🔧 初始化 SASRec 用于测速...")
    model_s = SASRec(config, test_data.dataset).to(config['device'])
    t_sasrec = measure_time(model_s, test_data, config['device'], "SASRec")
    
    try:
        print("🔧 初始化 K-GNN 用于测速...")
        n_entities = dataset.item_num 
        # 构造一个假的 KG 矩阵用于占位，保证模型能跑通
        kg_matrix = torch.zeros((n_entities, 5), dtype=torch.long).to(config['device'])
        model_k = Gated_KG_SASRec(config, test_data.dataset, kg_matrix, n_entities).to(config['device'])
        
        t_kgnn = measure_time(model_k, test_data, config['device'], "K-GNN")
        
        if t_sasrec > 0:
            overhead = (t_kgnn - t_sasrec) / t_sasrec * 100
            print(f"\n💡 结论: K-GNN 额外开销仅为 {overhead:.2f}%")
            
    except Exception as e:
        print(f"\n⚠️ 跳过 K-GNN 测试 (需要 model_kgnn.py): {e}")