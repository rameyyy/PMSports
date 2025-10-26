import ufc.models.build_df_create_predictions as prediction_handler
import ufc.models.simple_predictions as simple_preds
import ufc.bets.bets as bets
import ufc.bookmakers.update_accuracies as accuracy_update
from ufc.scrapes import *

def update_predictions_winners(conn):
    prediction_handler.update_predictions_with_results(conn)

def update_prediction_simplified(conn):
    last_two_event_urls = get_last_two_past_events(conn)
    event_ids_arr = [event_url.rstrip("/").split("/")[-1] for event_url in last_two_event_urls]
    if not event_ids_arr:
        return
    for event in event_ids_arr:
        simple_preds.update_prediction_simplified_with_results(conn, event)
        
def update_bets(conn):
    last_two_event_urls = get_last_two_past_events(conn)
    event_ids_arr = [event_url.rstrip("/").split("/")[-1] for event_url in last_two_event_urls]
    if not event_ids_arr:
        return
    for event in event_ids_arr:
        bets.update_bet_outcomes(conn, event)

def update_accuracies():
    accuracy_update.calculate_model_accuracies()
    
def make_predictions(conn):
    future_event_urls = get_future_event_urls(conn)
    event_ids_arr = [event_url.rstrip("/").split("/")[-1] for event_url in future_event_urls]
    if not event_ids_arr:
        return
    for event in event_ids_arr:
        prediction_handler.run(event)
        simple_preds.build_algopicks_rows(event)