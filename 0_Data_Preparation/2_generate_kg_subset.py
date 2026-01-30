# =============================================================================
# 🔗 Step 2: Generate KG Subset (Synced with .inter)
# 目的：根据 MOOCCubeX.inter 中实际出现的物品，提取对应的 KG 子集
# 修复：直接读取 .inter 文件，确保 KG 与冷启动截断后的数据 100% 对齐
# =============================================================================

import os
import pickle
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

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
# 读取刚才生成的 .inter 文件
INTER_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'MOOCCubeX', 'MOOCCubeX.inter')
# KG 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'MOOCCubeX_KG_Only')
KG_SOURCE = os.path.join(DATA_DIR, 'kg_data.pkl')

def generate_kg_subset():
    print(f"📂 Project Root: {PROJECT_ROOT}")
    print("🔗 Generating KG Subset synced with Interaction Data...")
    
    # 1. 读取 .inter 文件 (这是唯一的真理来源)
    if not os.path.exists(INTER_PATH):
        print(f"❌ 错误：找不到 {INTER_PATH}")
        print("   请先运行 0_preprocess_raw_data.py 生成截断后的数据。")
        return

    print(f"📖 读取交互数据: {INTER_PATH} ...")
    df_inter = pd.read_csv(INTER_PATH, sep='\t')
    
    # 获取所有活跃的 item_id
    # 列名通常是 'item_id:token'
    iid_col = [c for c in df_inter.columns if 'item_id' in c][0]
    active_items = set(df_inter[iid_col].unique())
    
    print(f"   📊 交互数据中包含 {len(active_items)} 个唯一物品")

    # 2. 加载原始 KG
    print(f"📦 加载原始 KG: {KG_SOURCE} ...")
    if not os.path.exists(KG_SOURCE):
        print(f"❌ 错误：找不到原始 KG 文件 {KG_SOURCE}")
        return

    with open(KG_SOURCE, 'rb') as f:
        raw_kg = pickle.load(f)
    
    # 兼容处理：有些 kg_data 是 dict, 有些直接是映射
    item2concept = raw_kg.get('item2concept', raw_kg) if isinstance(raw_kg, dict) else raw_kg
    
    # 3. 过滤 KG (只保留 active_items 里的)
    print("   ✂️ 正在过滤 KG...")
    
    filtered_kg = {}
    hit_count = 0
    
    def extract_id(s):
        # 从 '123' 或 'item_123' 中提取数字
        nums = re.findall(r'\d+', str(s))
        return int(nums[0]) if nums else None

    # 建立映射加速查找
    # 注意：.inter 里的 item_id 可能是 int 也可能是 str
    # 我们统一转为 int 进行比对
    active_ids_int = set()
    for iid in active_items:
        d = extract_id(iid)
        if d is not None: active_ids_int.add(d)

    for k, concepts in tqdm(item2concept.items(), desc="Filtering"):
        item_id = extract_id(k)
        if item_id in active_ids_int:
            filtered_kg[k] = concepts
            hit_count += 1
            
    print(f"   ✅ KG 过滤完成: 原有 {len(item2concept)} -> 现有 {len(filtered_kg)}")
    print(f"   📉 覆盖率: {hit_count / len(active_items) * 100:.2f}% 的物品拥有 KG 信息")

    # 4. 保存新的 KG 子集
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存 pkl (给模型加载用)
    # 为了兼容旧代码，我们更新原文件里的 item2concept 部分
    new_kg_data = {'item2concept': filtered_kg}
    # 如果原始数据里有别的表（如 concept2instruction），也可以选择保留或丢弃
    # 这里我们只保留最核心的 item2concept 以节省空间
    
    out_pkl = os.path.join(DATA_DIR, 'kg_data.pkl') # 注意：这里为了方便，我们其实不用覆盖原始数据，但模型读取路径通常是固定的
    # 为了不破坏原始数据，我们存到 dataset 目录下，但需要确认你的模型读取逻辑
    # 你的模型代码里写的是: kg_path = os.path.join(DATA_DIR, 'kg_data.pkl')
    # 所以我们得小心。
    
    # 💡 最佳实践：
    # 既然是 "Subset"，我们应该生成对应的 .kg 文件给 RecBole 用，
    # 或者生成一个新的 pkl 给你的模型用。
    # 你的模型代码 (3_exp...) 读取的是 DATA_DIR/kg_data.pkl。
    # 为了不影响其他实验，我们不要覆盖 data/kg_data.pkl。
    # 
    # 但是！你的 3_exp 代码是写死的读取 data/kg_data.pkl。
    # 为了逻辑闭环，我们这里还是**不覆盖**原始文件，而是依靠 .inter 文件的对齐。
    # 
    # 等等，如果 Script 2 只是为了生成 RecBole 格式的 KG 文件 (.kg, .link)，
    # 那么我们生成到 dataset/MOOCCubeX_KG_Only 下面即可。
    
    # 5. 生成 RecBole 标准 KG 文件 (可选，用于 KG 增强模型)
    # 格式: head_id:token    relation_id:token    tail_id:token
    kg_inter_file = os.path.join(OUTPUT_DIR, 'MOOCCubeX_KG_Only.kg')
    
    kg_triplets = []
    # 假设关系都是 "has_concept" (relation_id=1)
    for item, concepts in filtered_kg.items():
        iid = str(item)
        for c in concepts:
            kg_triplets.append([iid, '1', str(c)])
            
    df_kg = pd.DataFrame(kg_triplets, columns=['head_id:token', 'relation_id:token', 'tail_id:token'])
    df_kg.to_csv(kg_inter_file, sep='\t', index=False)
    print(f"   ✅ RecBole KG 文件已生成: {kg_inter_file}")
    
    # 6. 同时复制 .inter 文件到 KG_Only 目录，方便跑消融实验
    shutil.copy(INTER_PATH, os.path.join(OUTPUT_DIR, 'MOOCCubeX_KG_Only.inter'))
    print(f"   ✅ 已同步 .inter 文件到: {OUTPUT_DIR}")

import shutil

if __name__ == "__main__":
    generate_kg_subset()