from .utils import create_connection

def update_bet_outcomes(connection, event_id: str):
    """
    Update bet outcomes based on prediction_simplified results for a specific event.
    
    For bets with bet_outcome='pending' in the specified event:
    - If correct=1 in prediction_simplified, set bet_outcome='won'
    - If correct=0 in prediction_simplified, set bet_outcome='lost'
    - If correct=NULL in prediction_simplified, set bet_outcome='void'
    """
    
    print("\n" + "="*80)
    print(f"UPDATING BET OUTCOMES FOR EVENT: {event_id}")
    print("="*80)
    
    cursor = connection.cursor(dictionary=True)
    
    # Get all pending bets for this event with their prediction results
    query = """
    SELECT 
        b.fight_id,
        ps.correct
    FROM ufc.bets b
    JOIN ufc.prediction_simplified ps ON ps.fight_id = b.fight_id
    WHERE b.bet_outcome = 'pending'
      AND ps.event_id = %s;
    """
    
    cursor.execute(query, (event_id,))
    pending_bets = cursor.fetchall()
    
    if not pending_bets:
        print(f"\n✅ No pending bets to update for event: {event_id}")
        cursor.close()
        return {'won': 0, 'lost': 0, 'void': 0}
    
    print(f"\nFound {len(pending_bets)} pending bets to update\n")
    
    won_count = 0
    lost_count = 0
    void_count = 0
    
    for bet in pending_bets:
        fight_id = bet['fight_id']
        correct = bet['correct']
        
        # Determine new bet outcome
        if correct == 1:
            new_outcome = 'won'
            won_count += 1
            emoji = "✅"
        elif correct == 0:
            new_outcome = 'lost'
            lost_count += 1
            emoji = "❌"
        else:  # correct is NULL
            new_outcome = 'void'
            void_count += 1
            emoji = "⚪"
        
        # Update the bet
        update_query = """
        UPDATE ufc.bets
        SET bet_outcome = %s
        WHERE fight_id = %s AND bet_outcome = 'pending';
        """
        
        cursor.execute(update_query, (new_outcome, fight_id))
        print(f"   {emoji} Fight {fight_id}: {new_outcome}")
    
    # Commit all updates
    connection.commit()
    cursor.close()
    
    print("\n" + "="*80)
    print("UPDATE SUMMARY")
    print("="*80)
    print(f"✅ Won: {won_count} bets")
    print(f"❌ Lost: {lost_count} bets")
    print(f"⚪ Void: {void_count} bets")
    print(f"📊 Total updated: {won_count + lost_count + void_count} bets")
    print("="*80)
    
    return {
        'won': won_count,
        'lost': lost_count,
        'void': void_count
    }

def insert_bets_for_event(event_id: str, stake: float = 50.0):
    """
    Insert bets for all fights in an event based on AlgoPicks predictions.
    Only overwrites if potential_profit is better than existing bet.
    
    Args:
        event_id: The event_id to process
        stake: Amount to bet on each fight (default 50)
    """
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get event name from events table
        cursor.execute("""
            SELECT title as event_name, date as event_date
            FROM ufc.events
            WHERE event_id = %s
        """, (event_id,))
        
        event_info = cursor.fetchone()
        if not event_info:
            print(f"Event {event_id} not found in events table")
            return
        
        event_name = event_info['event_name']
        event_date = event_info['event_date']
        
        # Get all fights for this event from prediction_simplified
        cursor.execute("""
            SELECT 
                ps.fight_id,
                ps.fighter1_id,
                ps.fighter2_id,
                ps.fighter1_name,
                ps.fighter2_name,
                ps.algopick_prediction,
                ps.algopick_probability
            FROM ufc.prediction_simplified ps
            WHERE ps.event_id = %s
            AND ps.algopick_prediction IS NOT NULL
        """, (event_id,))
        
        predictions = cursor.fetchall()
        
        if not predictions:
            print(f"No predictions found for event {event_id}")
            return
        
        for pred in predictions:
            fight_id = pred['fight_id']
            fighter1_id = pred['fighter1_id']
            fighter2_id = pred['fighter2_id']
            fighter1_name = pred['fighter1_name']
            fighter2_name = pred['fighter2_name']
            algopick_pred = pred['algopick_prediction']
            algopick_prob = float(pred['algopick_probability'])
            
            # Determine which fighter AlgoPicks chose
            # 0 = fighter1, 1 = fighter2
            my_pick_id = fighter1_id if algopick_pred == 0 else fighter2_id
            my_pick_name = fighter1_name if algopick_pred == 0 else fighter2_name
            
            # fighter_bet_on: 0 if betting on fighter1, 1 if betting on fighter2
            fighter_bet_on = algopick_pred
            
            # Calculate predictions for both fighters based on algopick_probability
            if algopick_pred == 0:
                # Betting on fighter1
                fighter1_pred = algopick_prob
                fighter2_pred = 100 - algopick_prob
            else:
                # Betting on fighter2
                fighter1_pred = 100 - algopick_prob
                fighter2_pred = algopick_prob
            
            # Get odds for BOTH fighters from bookmaker_odds
            cursor.execute("""
                SELECT 
                    bookmaker,
                    fighter1_id,
                    fighter2_id,
                    fighter1_odds,
                    fighter2_odds,
                    fighter1_ev,
                    fighter2_ev,
                    fighter1_odds_percent,
                    fighter2_odds_percent
                FROM ufc.bookmaker_odds
                WHERE fight_id = %s
            """, (fight_id,))
            
            bookmaker_data = cursor.fetchall()
            
            if not bookmaker_data:
                print(f"No bookmaker odds found for fight_id {fight_id}, skipping...")
                continue
            
            # Find best odds for my pick across all bookmakers
            best_bet = None
            best_odds = -999999
            
            for book in bookmaker_data:
                # We need to map fighter IDs correctly
                # fighter1_id/fighter2_id from prediction_simplified should match bookmaker_odds
                
                # Check if the bookmaker has the same fighter order
                if book['fighter1_id'] == fighter1_id and book['fighter2_id'] == fighter2_id:
                    # Same order
                    book_fighter1_odds = book['fighter1_odds']
                    book_fighter1_ev = book['fighter1_ev']
                    book_fighter2_odds = book['fighter2_odds']
                    book_fighter2_ev = book['fighter2_ev']
                elif book['fighter1_id'] == fighter2_id and book['fighter2_id'] == fighter1_id:
                    # Swapped order - flip them
                    book_fighter1_odds = book['fighter2_odds']
                    book_fighter1_ev = book['fighter2_ev']
                    book_fighter2_odds = book['fighter1_odds']
                    book_fighter2_ev = book['fighter1_ev']
                else:
                    continue  # Fighter IDs don't match at all
                
                # Determine which fighter we're betting on and get their odds/ev
                if algopick_pred == 0:
                    # Betting on fighter1
                    my_fighter_odds = book_fighter1_odds
                    my_fighter_ev = book_fighter1_ev
                else:
                    # Betting on fighter2
                    my_fighter_odds = book_fighter2_odds
                    my_fighter_ev = book_fighter2_ev
                
                # Only consider if EV > 5
                if my_fighter_ev is not None and my_fighter_ev > 5:
                    if my_fighter_odds > best_odds:
                        best_odds = my_fighter_odds
                        best_bet = {
                            'bookmaker': book['bookmaker'],
                            'fighter1_odds': book_fighter1_odds,
                            'fighter1_ev': book_fighter1_ev,
                            'fighter2_odds': book_fighter2_odds,
                            'fighter2_ev': book_fighter2_ev,
                            'my_fighter_odds': my_fighter_odds
                        }
            
            # Insert bet if we found one with EV > 5
            if best_bet:
                # Calculate potential profit
                if best_bet['my_fighter_odds'] > 0:
                    potential_profit = stake * (best_bet['my_fighter_odds'] / 100)
                else:
                    potential_profit = stake * (100 / abs(best_bet['my_fighter_odds']))
                
                potential_loss = stake  # Just 50, not negative
                
                # Check if bet already exists for this fight_id
                cursor.execute("""
                    SELECT potential_profit
                    FROM ufc.bets
                    WHERE fight_id = %s
                """, (fight_id,))
                
                existing_bet = cursor.fetchone()
                
                # Only insert/overwrite if no existing bet OR new potential_profit is better
                if existing_bet is None or potential_profit > existing_bet['potential_profit']:
                    # Delete existing bet if it exists
                    if existing_bet:
                        cursor.execute("""
                            DELETE FROM ufc.bets
                            WHERE fight_id = %s
                        """, (fight_id,))
                    
                    # Insert new bet
                    cursor.execute("""
                        INSERT INTO ufc.bets (
                            bet_date,
                            bet_outcome,
                            bet_type,
                            event_id,
                            event_name,
                            fight_date,
                            fight_id,
                            fighter1_ev,
                            fighter1_name,
                            fighter1_odds,
                            fighter1_pred,
                            fighter2_ev,
                            fighter2_name,
                            fighter2_odds,
                            fighter2_pred,
                            fighter_bet_on,
                            potential_loss,
                            potential_profit,
                            sportsbook,
                            stake
                        ) VALUES (
                            CURDATE(),
                            'pending',
                            'moneyline',
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        event_id,
                        event_name,
                        event_date,
                        fight_id,
                        best_bet['fighter1_ev'],
                        fighter1_name,
                        str(best_bet['fighter1_odds']),
                        fighter1_pred,
                        best_bet['fighter2_ev'],
                        fighter2_name,
                        str(best_bet['fighter2_odds']),
                        fighter2_pred,
                        fighter_bet_on,
                        potential_loss,
                        potential_profit,
                        best_bet['bookmaker'],
                        stake
                    ))
                    
                    if existing_bet:
                        print(f"⟳ Updated bet for {fighter1_name} vs {fighter2_name}: {my_pick_name} at {best_bet['my_fighter_odds']} ({best_bet['bookmaker']}) - Better profit: ${existing_bet['potential_profit']:.2f} → ${potential_profit:.2f}")
                    else:
                        print(f"✓ Inserted bet for {fighter1_name} vs {fighter2_name}: {my_pick_name} at {best_bet['my_fighter_odds']} ({best_bet['bookmaker']}) - EV: {best_bet['fighter1_ev'] if algopick_pred == 0 else best_bet['fighter2_ev']}%")
                else:
                    print(f"⊘ Skipped {fighter1_name} vs {fighter2_name}: Existing bet has better profit (${existing_bet['potential_profit']:.2f} vs ${potential_profit:.2f})")
            else:
                print(f"✗ No qualifying bets (EV > 5) found for fight {fight_id}")
        
        conn.commit()
        print(f"\n✓ Successfully processed event {event_id}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error inserting bets: {e}")
        raise
    finally:
        cursor.close()
        conn.close()