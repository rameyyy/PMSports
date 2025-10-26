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