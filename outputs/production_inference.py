import pandas as pd
import joblib

def production_fraud_detection(raw_transactions, feature_engineer_path, model_path):
    """
    Complete production inference function

    Args:
        raw_transactions: DataFrame with columns [step, type, amount, nameOrig, 
                         oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, 
                         newbalanceDest, isFraud, isFlaggedFraud]
        feature_engineer_path: Path to saved feature engineer
        model_path: Path to saved model

    Returns:
        DataFrame with transaction_id, risk_score, alert
    """
    # Load components
    feature_eng = joblib.load(feature_engineer_path)
    model_config = joblib.load(model_path)
    model = model_config['model']
    threshold = model_config['optimal_threshold']

    # Transform features
    X_features = feature_eng.transform(raw_transactions)

    # Make predictions
    risk_scores = model.predict_proba(X_features)[:, 1]
    alerts = (risk_scores >= threshold).astype(int)

    # Create output
    results = pd.DataFrame({
        'transaction_id': raw_transactions.index,
        'risk_score': risk_scores,
        'alert': alerts,
        'threshold_used': threshold
    })

    return results
