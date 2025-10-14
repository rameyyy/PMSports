from models.ufc_mma import flat_df_build
from models.ufc_mma import model_development, chatgpt_press_analysis, attr_fighthist_combined_model, fight_features, fight_history_model, attributes_model
import polars as pl
df = pl.read_parquet('fight_features_extracted.parquet')
x = attr_fighthist_combined_model.run(df)
print(x)
exit()
df = pl.read_parquet('fight_features_extracted.parquet')
### TODO: Make a file in models that can select all upcoming fights, build a flat df. build the fight features df from it, pass that df into attr and fight hist combined
### file, then get the amount of fights and the amount of coverage,, then push that fight_id with the predictions to sql for upcoming events, combined accuracy = ~61.68%
### fight_hist_avges = 59.57%, attributes avg = 59.96% accuracy