#!/bin/bash

# =============================================================================
# 🚀 K-GNN Experiments: One-Click Reproduction Script
# =============================================================================
# 强制 Python 使用 UTF-8，防止 Windows 上打印 Emoji 报错
export PYTHONIOENCODING=utf-8
# 如果任何一行命令失败，立即停止脚本
set -e

echo "====================================================================="
echo "▶️  Step 0: Checking Environment..."
echo "====================================================================="
# 简单检查是否安装了核心包
python -c "import recbole; import ray; import torch; print('✅ Environment checks passed.')"

echo ""
echo "====================================================================="
echo "▶️  Step 1: Data Preparation"
echo "====================================================================="
echo "   [1/3] Preprocessing raw data..."
python 0_Data_Preparation/0_preprocess_raw_data.py

echo "   [2/3] Generating dataset statistics (Table 1)..."
python 0_Data_Preparation/1_dataset_statistics.py

echo "   [3/3] Generating KG subset..."
python 0_Data_Preparation/2_generate_kg_subset.py

echo ""
echo "====================================================================="
echo "▶️  Step 2: Main Experiments (Performance Comparison)"
echo "====================================================================="
echo "   [1/2] Running Main Comparison (Table 3)..."
python 1_Main_Experiments/3_exp_extreme_cold_start.py

echo "   [2/2] Running Sparsity Robustness Test (Figure 2 / Table 4)..."
python 1_Main_Experiments/4_exp_data_sparsity.py

echo ""
echo "====================================================================="
echo "▶️  Step 3: Ablation & Analysis"
echo "====================================================================="
echo "   [1/5] Running Long-tail Diagnosis..."
python 2_Ablation_and_Analysis/5_analysis_long_tail.py

echo "   [2/5] Running Ablation Study (Table 5)..."
python 2_Ablation_and_Analysis/6_exp_ablation_study.py

echo "   [3/5] Running Parameter Sensitivity (Figure 5)..."
python 2_Ablation_and_Analysis/7_exp_parameter_sensitivity.py

echo "   [4/5] Generating Case Study (Figure 6)..."
python 2_Ablation_and_Analysis/8_case_study_visualization.py

echo "   [5/5] Testing Inference Latency (Section 4.5.2)..."
python 2_Ablation_and_Analysis/9_test_inference_time.py

echo ""
echo "====================================================================="
echo "✅ All experiments completed successfully!"
echo "   Results are saved in the logs and generated figures."
echo "====================================================================="