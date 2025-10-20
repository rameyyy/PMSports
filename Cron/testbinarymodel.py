from models.ufc_mma import build_df_create_predictions, tempdelme
# build_df_create_predictions.run('130325-ufc-321')
rt = tempdelme.get_model_accuracy_at_prob('ensemble_weightedavgprob', 0.43191101)
print(rt)