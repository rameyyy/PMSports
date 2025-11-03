from update_ufc_db import *
from update_or_predict import *
from scrapes import create_connection
import time

if __name__ == "__main__":
    start_time = time.time()
    conn = create_connection()
    print('Beginning UFC Cron processes')

    ### CRON SCRAPE PROCESSES ###
    # get_new_upcoming_events(conn=conn)
    # update_scrapes_for_upcoming_events(conn=conn)
    # update_last2_events_outcomes(conn=conn)
    #############################

    ### ML MODEL PROCESSES ###
    # update_predictions_winners(conn=conn)
    # update_prediction_simplified(conn=conn)
    # update_accuracies()
    # update_bets(conn=conn)
    # make_predictions(conn=conn)
    # update_bookmakers()
    # make_bets_upcoming_events(conn=conn)
    update_bet_analytics(conn=conn)
    ##########################

    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"UFC Cron took {minutes} minutes and {seconds:.2f} seconds")
