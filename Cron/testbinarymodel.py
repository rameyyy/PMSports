from models.ufc_mma import get_accuracies
from models.ufc_mma import build_df_create_predictions, simple_predictions
# build_df_create_predictions.run('130325-ufc-321')
# rt = get_accuracies.get_model_accuracy_at_prob('ensemble_weightedavgprob', 0.43191101)
acc= get_accuracies.get_all_model_accuracies(False)
print(acc)
df = simple_predictions.build_algopicks_rows(include_legacy_for_model_choice=False)
simple_predictions.push_algopicks_to_sql(df)
# after you build the df:
# df = simple_predictions.build_algopicks_rows(include_legacy_for_model_choice=False)

# You already have:
# df = simple_predictions.build_algopicks_rows(include_legacy_for_model_choice=False)

# df = simple_predictions.build_algopicks_rows(include_legacy_for_model_choice=False)
# df.write_csv("simplepred.csv")
df.write_csv('simplepred.csv')
import polars as pl

# Filter only known results
# Filter to valid rows (where correct is not null means both actual_winner and prediction exist)
df_valid = df.filter(pl.col("correct").is_not_null())

# Count totals
total_rows = df.height
valid = df_valid.height
correct = int(df_valid["correct"].sum())

accuracy = correct / valid if valid > 0 else None

print(f"Total rows: {total_rows}")
print(f"Valid predictions (with known outcomes): {valid}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.4f}" if accuracy is not None else "Accuracy: N/A")
