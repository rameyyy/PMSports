from .utils import create_connection, fetch_query
from typing import List


def get_model_accuracies_batched(model: str, probs: List[float], window: float = 0.01) -> dict:
    """
    Get accuracies for multiple probability values in one query.
    Returns dict: {prob: {"correct": int, "prob_range": int}}
    """
    # Build CASE statements for each probability band
    cases = []
    for i, prob in enumerate(probs):
        lower = prob - window
        upper = prob + window
        cases.append(f"""
            SUM(CASE 
                WHEN {model}_f1_prob BETWEEN {lower} AND {upper} 
                     AND {model}_correct IS NOT NULL 
                THEN 1 ELSE 0 
            END) AS band_{i}_total,
            SUM(CASE 
                WHEN {model}_f1_prob BETWEEN {lower} AND {upper} 
                     AND {model}_correct = 1 
                THEN 1 ELSE 0 
            END) AS band_{i}_correct
        """)
    
    query = f"""
    SELECT {','.join(cases)}
    FROM ufc.predictions
    WHERE {model}_f1_prob IS NOT NULL;
    """
    
    conn = create_connection()
    df = fetch_query(conn, query)
    
    if not df:
        return {prob: {"correct": 0, "prob_range": 0} for prob in probs}
    
    result = {}
    for i, prob in enumerate(probs):
        total = int(df[0].get(f"band_{i}_total", 0))
        correct = int(df[0].get(f"band_{i}_correct", 0))
        result[prob] = {"correct": correct, "prob_range": total}
    
    return result

def get_all_model_accuracies(include_legacy: bool = True) -> dict:
    """
    Retrieves accuracy from the ufc.accuracies table for specified models.
    """
    conn = create_connection()

    models = [
        "logistic",
        "xgboost",
        "gradient",
        "ensemble_avgprob",
        "ensemble_weightedavgprob",
    ]

    # Convert model names to match table format
    model_mapping = {
        "logistic": "Logistic",
        "xgboost": "XGBoost",
        "gradient": "Gradient",
        "ensemble_avgprob": "Ensemble Avg Prob",
        "ensemble_weightedavgprob": "Ensemble Weight Avg Prob",
    }

    # Build list of model names for the query
    table_model_names = [model_mapping.get(m, m) for m in models]
    placeholders = ','.join(['%s'] * len(table_model_names))

    query = f"""
    SELECT model_name, accuracy 
    FROM ufc.model_accuracies
    WHERE model_name IN ({placeholders})
    """

    df = fetch_query(conn, query, tuple(table_model_names))

    if not df:
        return {}

    # Create result dict, mapping back to original model names
    reverse_mapping = {v: k for k, v in model_mapping.items()}
    accuracies = {}
    for row in df:
        table_name = row['model_name']
        original_name = reverse_mapping.get(table_name, table_name.lower().replace(' ', '_'))
        if original_name in models:
            accuracies[original_name] = round(float(row['accuracy']), 4)

    return accuracies