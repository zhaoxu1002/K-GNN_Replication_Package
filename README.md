# Replication Package for K-GNN

This repository contains the source code and experimental scripts for the paper: **"Time-Aware Neural Collaborative Filtering Framework with Hybrid Feedback for Micro-Course Recommendation"**.

The proposed model, **K-GNN**, integrates Knowledge Graph (KG) information via a Gated Late-Fusion mechanism to alleviate the cold-start problem in micro-course recommendation.

## 📂 Project Structure

The project is organized into three main modules corresponding to the experimental workflow:

```text
.
├── 0_Data_Preparation/
│   ├── 0_preprocess_raw_data.py    # Data cleaning & cold-start truncation
│   ├── 1_dataset_statistics.py     # Generate statistics for Table 1
│   └── 2_generate_kg_subset.py     # Align KG data with interaction data
│
├── 1_Main_Experiments/
│   ├── 3_exp_extreme_cold_start.py # Main Comparison (Table 3)
│   └── 4_exp_data_sparsity.py      # Robustness Analysis (Figure 2 / Table 4)
│
├── 2_Ablation_and_Analysis/
│   ├── 5_analysis_long_tail.py     # Long-tail distribution diagnosis
│   ├── 6_exp_ablation_study.py     # Ablation Study (Table 5)
│   ├── 7_exp_parameter_sensitivity.py # Hyperparameter Sensitivity (Figure 5)
│   ├── 8_case_study_visualization.py  # Case Study (Figure 6)
│   └── 9_test_inference_time.py       # Efficiency Test (Section 4.5.2)
│
├── src/                            # Core model implementation
│   └── model_kgnn.py               # Gated_KG_SASRec model class
│
├── data/                           # Raw data files (pkl, csv)
├── dataset/                        # Processed RecBole atomic files
├── saved/                          # Trained model checkpoints
├── All_Experiments.ipynb           # Jupyter Notebook with full execution logs 
├── run_all.sh                      # One-click execution script
└── requirements.txt                # Python dependencies

💻 Environment & Requirements
The experiments were conducted on a Linux server with the following specifications:

Date: Jan 23, 2026

GPU: 1x NVIDIA A10 (24GB VRAM)

Driver Version: 550.54.15

CUDA Version: 12.4

Dependencies
The project relies on RecBole (v1.1.1) and auxiliary libraries. You can install the necessary environment using the provided requirements file:
pip install -r requirements.txt

Alternatively, you can manually install the key packages:
pip install recbole ray kmeans_pytorch matplotlib
    Note: The scripts contain compatibility patches for Numpy 2.0. The code will automatically handle np.float_ / np.int_ mappings.

🚀 Quick Start (Replication Steps)
You can run the experiments in three ways:

Method A: One-Click Execution (Recommended for Linux/Mac)
We provide a shell script to run the full pipeline sequentially:
chmod +x run_all.sh
./run_all.sh

Method B: Jupyter Notebook (Recommended for Review)
Open All_Experiments.ipynb.

View Results: You can view the pre-executed outputs and logs directly in this notebook without running any code.

Reproduce: You can also run the cells step-by-step to reproduce the experiments interactively.

Method C: Manual Execution
Please run the scripts in the following order from the root directory of the project:

Step 1: Data Preparation
Preprocess the raw MOOCCubeX data and generate the dataset statistics.

python 0_Data_Preparation/0_preprocess_raw_data.py
python 0_Data_Preparation/1_dataset_statistics.py
python 0_Data_Preparation/2_generate_kg_subset.py

Step 2: Main Performance Experiments
Run the core comparison (SASRec vs. K-GNN) and sparsity robustness tests.
# Generates results for Table 3
python 1_Main_Experiments/3_exp_extreme_cold_start.py

# Generates results for Figure 2 and Table 4
python 1_Main_Experiments/4_exp_data_sparsity.py

Step 3: Analysis & Visualization
Run ablation studies, sensitivity analysis, and efficiency tests.
# Ablation Study (Table 5)
python 2_Ablation_and_Analysis/6_exp_ablation_study.py

# Parameter Sensitivity (Figure 5)
python 2_Ablation_and_Analysis/7_exp_parameter_sensitivity.py

# Inference Efficiency (Section 4.5.2)
python 2_Ablation_and_Analysis/9_test_inference_time.py

📊 Expected Results

Script	                         Corresponds to Paper	         Description
3_exp_extreme_cold_start.py	      Table 3	                     Compares K-GNN against baselines on metrics (NDCG, Hit, Recall).
4_exp_data_sparsity.py	          Figure 2	                     Evaluates performance under varying data sparsity ratios (20% - 100%).
6_exp_ablation_study.py	          Table 5	                     Validates the effectiveness of the Gated Late-Fusion mechanism.
7_exp_parameter_sensitivity.py	  Figure 5	                     Analyzes the impact of different embedding sizes (32, 64, 128).
9_test_inference_time.py	      Section 4.5.2	                 Demonstrates that K-GNN incurs negligible inference overhead (<0.01%).

📧 Contact
If you have any questions about the code or the paper, please refer to the author contact information in the manuscript.