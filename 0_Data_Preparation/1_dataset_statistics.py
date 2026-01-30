# =============================================================================
# 📊 Table 1: Dataset Statistics (Synced with Cold-Start Experiment)
# 目的：生成论文 Table 1 的统计数据
# 修正：读取实际用于实验的 .inter 文件，而不是原始 CSV
# =============================================================================

import os
import pandas as pd
import numpy as np

def setup_paths():
    try: current_path = os.path.dirname(os.path.abspath(__file__))
    except: current_path = os.getcwd()
    check_path = current_path
    for _ in range(3):
        if os.path.exists(os.path.join(check_path, 'data')): return check_path
        parent = os.path.dirname(check_path)
        if parent == check_path: break
        check_path = parent
    return current_path

PROJECT_ROOT = setup_paths()
# 🔥 关键修改：读取 .inter 文件 (制表符分隔)
DATA_FILE = os.path.join(PROJECT_ROOT, 'dataset', 'MOOCCubeX', 'MOOCCubeX.inter')

def print_statistics():
    print(f"📂 路径校准成功:")
    print(f"   Root: {PROJECT_ROOT}")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ 错误：找不到实验数据 {DATA_FILE}")
        print("   请先运行 0_preprocess_raw_data.py 生成截断数据。")
        return

    print(f"📖 正在读取实验数据 (Final Cold-Start Data): {DATA_FILE} ...")
    
    # RecBole 的 .inter 文件通常是 tab 分隔
    df = pd.read_csv(DATA_FILE, sep='\t')
    
    # 获取列名 (user_id:token, item_id:token, ...)
    uid_col = [c for c in df.columns if 'user_id' in c][0]
    iid_col = [c for c in df.columns if 'item_id' in c][0]
    
    n_users = df[uid_col].nunique()
    n_items = df[iid_col].nunique()
    n_inter = len(df)
    
    # 计算稀疏度
    # Sparsity = 1 - (Interactions / (Users * Items))
    sparsity = 1 - (n_inter / (n_users * n_items))
    
    # 计算平均交互数
    avg_actions = n_inter / n_users

    print("\n" + "="*40)
    print("✅ Table 1 Statistics (Cold-Start Subset)")
    print("="*40)
    print(f"# Users:        {n_users}")
    print(f"# Items:        {n_items}")
    print(f"# Interactions: {n_inter}")
    print(f"Avg. Actions:   {avg_actions:.2f}")
    print(f"Sparsity:       {sparsity:.4%}")
    print("="*40)
    
    if avg_actions > 15:
        print("⚠️ 警告：平均交互数过高，这看起来不像冷启动数据！")
    else:
        print("✅ 确认：平均交互数较低，符合冷启动实验设定。")

if __name__ == "__main__":
    print_statistics()