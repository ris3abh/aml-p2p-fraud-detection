## P2P Fraud Detection - Technical Documentation

#### Executive Summary

Built machine learning system for suspicious activity detection in peer-to-peer transactions using PaySim dataset (6.36M transactions). Achieved 26.1% SAR conversion rate at 2% alert capacity, significantly exceeding industry benchmarks (10-15%) despite extreme class imbalance (1:974) and 11x temporal concept drift.
Problem Analysis

#### Dataset Characteristics:

6,362,620 transactions over 30 days (744 hourly steps)
Extreme imbalance: 8,213 frauds (0.129%)
Temporal drift: Train 0.103% → Test 1.142% fraud rate (11.1x increase)
Critical constraint: Balance columns excluded due to label leakage

#### Key Discovery: 

The initial model achieved a suspicious AUPRC of 0.988. Investigation revealed that balance-derived features created perfect separation between fraud/normal transactions because PaySim's simulator cancels fraudulent transactions. Removed all balance features and rebuilt with legitimate patterns only.
Solution Architecture

#### Feature Engineering (24 features):

Temporal: Fraud peaks at hours 3-6 AM (146x lift), cyclical encoding for time patterns
Amount: Z-scores by type, high amounts (>$1M), round numbers ($1k, $10k = 70x lift)
Behavioral: Risky types (TRANSFER/CASH_OUT only types with fraud)
Key interaction: risky_type × fraud_peak_hour = 681x fraud lift

#### Model Selection:

Tested XGBoost (AUPRC 0.341), LightGBM (failed), CatBoost (0.380), ARF (failed)
Selected CatBoost: 11.6% improvement over XGBoost, handles extreme imbalance better
Applied scale_pos_weight=974 for cost-sensitive learning

#### Probability Calibration:

Isotonic regression reduced Brier score from 0.237 to 0.0086 (27x improvement)
Enables risk-based decisions and regulatory compliance

## Results

#### Model Performance:

AUPRC: 0.380 (33x better than random baseline)
AUC-ROC: 0.908
Top decile lift: 7.0x

#### Business Impact (2% alert budget = 3,246 transactions):

SAR conversion: 26.1% (industry best-in-class: 15-30%)
Recall: 49.4% (916/1,854 frauds caught)
Precision: 0.259
Annual savings: ~$2.1M vs. alerting all transactions

Alternative scenario (5% budget): 13.0% SAR conversion, 57.2% recall

#### Implementation Details
Temporal split: Train on days 1-24, test on days 25-31 (realistic production scenario)
Threshold optimization: Cost function (FN=$500, FP=$5) yields threshold 0.025 (operationally infeasible). Business-constrained threshold 0.987 balances detection with analyst capacity.

#### Code structure:

notebooks/01_eda.ipynb: Data exploration and leakage discovery
notebooks/02_feature_engineering.ipynb: Clean feature creation
notebooks/03_baseline_models.ipynb: Model training and evaluation
outputs/risk_scored_holdout_final.csv: Scored test set with calibrated probabilities

#### Key Insights

Data quality > Model complexity: Identifying and removing leakage was more impactful than algorithm choice
Business constraints > Statistical optima: 2% alert budget constraint overrides cost-optimal threshold
Simple features dominate: Transaction type × time interaction (681x lift) outperformed complex engineering
Calibration critical: Raw probabilities unreliable; isotonic regression enables business decisions

Repository: https://www.github.com/ris3abh/aml-p2p-fraud-detection
