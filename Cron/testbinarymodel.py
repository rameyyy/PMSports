import polars as pl
from models.ufc_mma import binary_model

# Step 1: Load your fight snapshots data
differential_df = pl.read_csv('trainingset.csv')  # or however you load it

# Step 2: Create differential features
# print("Creating differential features...")
# differential_df = flat_df_to_model_ready.create_differential_features(df)

# Step 3: Train and evaluate the model
print("\nTraining model...")
results = binary_model.train_and_evaluate_model(
    differential_df, 
    model_type='xgboost',  # Options: 'logistic', 'xgboost', 'gradient_boost'
    test_split=0.25         # 20% most recent fights for testing
)

# Step 4: View results
print(f"\n✅ Model trained successfully!")
print(f"Test Accuracy: {results['test_accuracy']:.2%}")

# Optional: See wrong predictions
wrong_predictions = results['test_results'].filter(pl.col('correct') == False)
print(f"\nWrong predictions: {len(wrong_predictions)}")
print(wrong_predictions.head(10))

# Optional: Save the trained model
import pickle
with open('mma_model.pkl', 'wb') as f:
    pickle.dump(results['model'], f)
with open('mma_scaler.pkl', 'wb') as f:
    pickle.dump(results['scaler'], f)
print("\n✅ Model saved to mma_model.pkl")