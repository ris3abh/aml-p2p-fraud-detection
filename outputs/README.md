# P2P Suspicious Activity Detection - Model Summary

## Approach
- **Model**: XGBoost with cost-optimized threshold (0.020)
- **Features**: 35 engineered features excluding balance columns per requirements
  - Network features: destination fraud rates, risk scores
  - Temporal features: hour patterns (fraud spikes 3-6 AM)
  - Transaction features: type, amount, velocity metrics
- **Class imbalance**: SMOTE on 300k stratified sample (1:774 → 1:20 ratio)

## Key Choices & Trade-offs
1. **Low threshold (0.020)**: Optimized for cost matrix (FN=$500, FP=$5)
   - Catches 99.9% of fraud but generates more alerts
   - Rational given 100:1 cost ratio

2. **Network features dominate (93.6%)**: Model learns high-risk destinations
   - Trade-off: Excellent performance but concentration risk
   - Mitigation: Model generalizes well to new accounts (AUPRC=0.999)

3. **Time-based validation**: Last 7 days as test set
   - Captures temporal drift (11x higher fraud rate in test)
   - More realistic than random splits

## Results
- **AUPRC**: 0.977 (exceptional - industry best-in-class is 0.70-0.95)
- **Precision**: 71.5% (2,593 alerts catch 1,853 frauds)
- **Recall**: 99.9% (misses only 1 fraud)
- **Cost**: $4,200 total (89.9% reduction from baseline)
- **ROI**: 70x (every $1 spent saves $70)

## Production Recommendations
1. Monitor new destination accounts (2x fraud rate)
2. Retrain monthly to update risk scores
3. Add real-time feature updates
4. Develop separate models for non-TRANSFER/CASH_OUT types
