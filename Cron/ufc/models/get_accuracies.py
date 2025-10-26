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
    Computes accuracy (correct / total) for every *_correct column
    in the ufc.predictions table.
    """
    conn = create_connection()

    models = [
        "logistic",
        "xgboost",
        "gradient",
        "homemade",
        "ensemble_majorityvote",
        "ensemble_weightedvote",
        "ensemble_avgprob",
        "ensemble_weightedavgprob",
    ]

    select_exprs = ",\n".join(
        [f"AVG(CASE WHEN p.{m}_correct = 1 THEN 1.0 ELSE 0 END) AS {m}_accuracy" for m in models]
    )

    if not include_legacy:
        where_clause = "WHERE p.legacy = 0 AND p.actual_winner IS NOT NULL AND f.fight_date > '2025-10-03'"
    else:
        where_clause = "WHERE p.actual_winner IS NOT NULL AND f.fight_date > '2025-10-03'"
    
    query = f"""
    SELECT {select_exprs} 
    FROM ufc.predictions p
    JOIN ufc.fights f ON f.fight_id = p.fight_id
    {where_clause};
    """

    df = fetch_query(conn, query)

    if not df:
        return {}

    row = df[0]
    accuracies = {m: round(float(row[f"{m}_accuracy"]), 4) for m in models}

    return accuracies