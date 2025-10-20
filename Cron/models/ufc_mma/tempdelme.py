from .utils import create_connection, fetch_query  # your existing SQL fetch helper

def get_model_accuracy_at_prob(model: str, prob: float, window: float = 0.01) -> dict:
    """
    Calculates actual accuracy for a given model around a target probability.
    Example:
        get_model_accuracy_at_prob("logistic", 0.25)
    
    Parameters
    ----------
    model : str
        One of 'logistic', 'xgboost', 'gradient', etc.
    prob : float
        Target probability (e.g. 0.25)
    window : float
        Range around prob (default ±0.01)

    Returns
    -------
    dict
        {
          "model": "logistic",
          "prob_range": [0.24, 0.26],
          "sample_size": 215,
          "empirical_accuracy": 0.238,
        }
    """

    lower = prob - window
    upper = prob + window

    query = f"""
    SELECT 
        COUNT(*) AS total_rows,
        SUM(CASE WHEN {model}_correct = 1 THEN 1 ELSE 0 END) AS correct_rows
    FROM ufc.predictions
    WHERE {model}_f1_prob BETWEEN {lower} AND {upper}
      AND {model}_correct IS NOT NULL;
    """
    conn = create_connection()
    df = fetch_query(conn, query)
    print(df)
    total = int(df[0, "total_rows"])
    correct = int(df[0, "correct_rows"]) if df[0, "correct_rows"] is not None else 0

    accuracy = round(correct / total, 4) if total > 0 else None

    return {
        "model": model,
        "prob_range": [round(lower, 2), round(upper, 2)],
        "sample_size": total,
        "empirical_accuracy": accuracy,
    }
