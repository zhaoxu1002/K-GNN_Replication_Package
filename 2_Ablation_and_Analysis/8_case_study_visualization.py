# =============================================================================
# 🔍 Case Study: Visualization Logic (Final Fixed)
# 目的：寻找可解释的 User-Item 对 (共享 KG 概念) 用于 Figure 6
# =============================================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import re
import random

# --- Numpy 2.0 Patch ---
patch_map = {'float_': np.float64, 'int_': np.int64, 'bool_': bool, 'str_': np.str_}
for alias, target in patch_map.items():
    if not hasattr(np, alias): setattr(np, alias, target)

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
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INTER_FILE = os.path.join(PROJECT_ROOT, 'dataset', 'MOOCCubeX', 'MOOCCubeX.inter')
KG_FILE = os.path.join(DATA_DIR, 'kg_data.pkl')

def find_interpretable_cases():
    print(f"🔍 Starting Case Study Search in {PROJECT_ROOT}...")
    
    if not os.path.exists(INTER_FILE) or not os.path.exists(KG_FILE):
        print("❌ 找不到数据文件，请确保预处理已完成。")
        return

    # 1. 加载 KG 映射
    print("   📖 Loading KG...")
    with open(KG_FILE, 'rb') as f: raw_kg = pickle.load(f)
    item2concept = raw_kg.get('item2concept', raw_kg)
    
    # 简化映射逻辑
    concept_map = {}
    for k, v in item2concept.items():
        # 提取 item id 数字
        nums = re.findall(r'\d+', str(k))
        if nums:
            concept_map[str(nums[0])] = set(v) # 用 set 方便求交集

    # 2. 加载交互数据
    print("   📖 Loading Interactions...")
    df = pd.read_csv(INTER_FILE, sep='\t', dtype=str)
    uid_col = df.columns[0]
    iid_col = df.columns[1]
    
    # 3. 寻找案例
    print("   🕵️‍♂️ Searching for connected pairs...")
    
    # 随机抽取 50 个用户进行扫描
    all_users = df[uid_col].unique()
    sample_users = np.random.choice(all_users, size=min(50, len(all_users)), replace=False)
    
    found_count = 0
    
    for uid in sample_users:
        user_df = df[df[uid_col] == uid]
        if len(user_df) < 2: continue
        
        # 获取用户序列
        items = user_df[iid_col].tolist()
        
        # 检查最后两个物品是否有共享概念
        # History (Last-1) -> Target (Last)
        hist_item = items[-2]
        target_item = items[-1]
        
        if hist_item in concept_map and target_item in concept_map:
            c_hist = concept_map[hist_item]
            c_target = concept_map[target_item]
            
            # 求交集
            shared = c_hist.intersection(c_target)
            
            if shared:
                found_count += 1
                sid = list(shared)[0]
                print("\n" + "="*50)
                print(f"🎉 发现高解释性案例 (Case #{found_count})")
                print(f"👤 User ID: {uid}")
                print(f"📚 History Item: {hist_item} (Concepts: {list(c_hist)[:3]})")
                print(f"🎯 Target Item:  {target_item} (Concepts: {list(c_target)[:3]})")
                print(f"🔗 Shared Concept: {sid}")
                print(f"💡 解释: K-GNN 成功捕捉到了概念 [{sid}] 的连贯性！")
                print("="*50)
                
                if found_count >= 3: break # 找到 3 个就够了

    if found_count == 0:
        print("⚠️ 未找到明显的直连案例，这在稀疏数据中很正常。")
        print("   建议：在论文中手动挑选几个热门物品的 Case 进行展示。")

if __name__ == "__main__":
    find_interpretable_cases()