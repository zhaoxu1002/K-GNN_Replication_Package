# =============================================================================
# 🧹 Step 0: Preprocessing (The Mixed Data Strategy)
# 核心改动：不再过滤无 KG 的物品！保留噪音，增加难度。
# =============================================================================

import os
import pandas as pd
import numpy as np
import re

def setup_project_paths():
    try: current_path = os.path.dirname(os.path.abspath(__file__))
    except: current_path = os.getcwd()
    check_path = current_path
    for _ in range(3):
        if os.path.exists(os.path.join(check_path, 'data')): return check_path
        parent = os.path.dirname(check_path)
        if parent == check_path: break
        check_path = parent
    return current_path

PROJECT_ROOT = setup_project_paths()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'MOOCCubeX')
INPUT_CSV = os.path.join(DATA_DIR, 'mooccubex_cleaned_data.csv')
OUTPUT_INTER = os.path.join(DATASET_DIR, 'MOOCCubeX.inter')

def preprocess_mixed_data():
    print(f"📂 Project Root: {PROJECT_ROOT}")
    print(f"📖 读取交互 CSV: {INPUT_CSV} ...")
    
    # 1. 读取原始数据
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    
    # 2. 识别列名
    cols = df.columns
    uid_col = next((c for c in cols if c in ['user_id', 'userId']), cols[0])
    iid_col = next((c for c in cols if c in ['item_id', 'problem_id']), cols[1])
    ts_col = next((c for c in cols if c in ['timestamp', 'time', 'submit_time']), cols[-1])
    
    print(f"   📊 原始交互数: {len(df)}")

    # 3. ID 规整化 (统一转为纯数字字符串，方便后续匹配，但不过滤)
    print("   🔧 规整化 Item ID (提取数字)...")
    def extract_id_from_csv(x):
        nums = re.findall(r'\d+', str(x))
        return nums[0] if nums else str(x) # 找不到数字就保留原样
    
    df[iid_col] = df[iid_col].apply(extract_id_from_csv)
    
    # 🔥 关键点：这里不再加载 KG 进行过滤！我们保留所有物品！ 🔥
    # 这样 SASRec 就要面对大量它“看不懂”的冷门物品了。
    
    # 4. 冷启动截断 (保留前 8 条)
    print("   ⏳ 按时间排序并执行【True Cold-Start】截断 (Top 8)...")
    try: df[ts_col] = pd.to_numeric(df[ts_col], errors='raise')
    except: df[ts_col] = pd.to_datetime(df[ts_col]).astype('int64') // 10**9
    
    # 去重
    df.drop_duplicates(subset=[uid_col, iid_col], keep='first', inplace=True)
    
    # 排序
    df = df.sort_values(by=[uid_col, ts_col])
    
    # 截断
    KEEP_N = 8
    df_cold = df.groupby(uid_col).head(KEEP_N).reset_index(drop=True)
    
    # 过滤过短用户 (至少 5 条，保证基本训练)
    user_counts = df_cold[uid_col].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df_cold = df_cold[df_cold[uid_col].isin(valid_users)]
    
    print(f"   📉 最终行数: {len(df_cold)}")
    print(f"   👥 剩余用户: {df_cold[uid_col].nunique()}")
    print(f"   📦 剩余物品: {df_cold[iid_col].nunique()} (包含无 KG 物品)")

    # 5. 保存
    os.makedirs(os.path.dirname(OUTPUT_INTER), exist_ok=True)
    df_inter = df_cold[[uid_col, iid_col, ts_col]].copy()
    df_inter.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
    
    df_inter.to_csv(OUTPUT_INTER, sep='\t', index=False)
    print(f"   ✅ 混合数据集生成完毕: {OUTPUT_INTER}")

if __name__ == "__main__":
    preprocess_mixed_data()