from .utils import create_connection, fetch_query
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class OddsProcessor:
    def __init__(self, conn, json_file: str = "ufc_odds.json"):
        self.conn = conn
        self.json_file = json_file
        
    def load_odds_data(self) -> List[Dict]:
        """Load odds data from JSON file"""
        with open(self.json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def find_fight_by_names(self, name1: str, name2: str, event_date: str) -> Optional[Dict]:
        """Find fight in prediction_simplified by fighter names and date
        Returns the full fight row if found
        """
        
        def normalize_name(name: str) -> str:
            """Normalize name by removing spaces and converting to lowercase"""
            return name.replace(" ", "").replace("-", "").lower()
        
        def get_name_variations(name: str) -> list:
            """Generate variations of a name for matching"""
            variations = [f"%{name}%"]  # Full name
            parts = name.split()
            
            if len(parts) > 1:
                # Last name only
                variations.append(f"%{parts[-1]}%")
                # First name only
                variations.append(f"%{parts[0]}%")
                # Concatenated (for "Jun Yong" -> "Junyong")
                variations.append(f"%{''.join(parts)}%")
                # First + Last without middle
                if len(parts) > 2:
                    variations.append(f"%{parts[0]}%{parts[-1]}%")
            
            return variations
        
        # Get variations for both names
        name1_vars = get_name_variations(name1)
        name2_vars = get_name_variations(name2)
        
        # Try progressively broader searches
        base_query = """
            SELECT fight_id, fighter1_id, fighter2_id, fighter1_name, fighter2_name,
                algopick_prediction, algopick_probability, date
            FROM ufc.prediction_simplified 
            WHERE {conditions}
            AND date >= DATE_SUB(%s, INTERVAL 1 DAY)
            AND date <= DATE_ADD(%s, INTERVAL 1 DAY)
            LIMIT 1
        """
        
        # Strategy 1: Try each variation combination
        for n1_var in name1_vars:
            for n2_var in name2_vars:
                conditions = """
                    ((fighter1_name LIKE %s AND fighter2_name LIKE %s)
                    OR (fighter1_name LIKE %s AND fighter2_name LIKE %s))
                """
                query = base_query.format(conditions=conditions)
                
                result = fetch_query(self.conn, query, 
                                (n1_var, n2_var, n2_var, n1_var, event_date, event_date))
                
                if result and len(result) > 0:
                    row = result[0]
                    if isinstance(row, dict):
                        return {
                            'fight_id': row['fight_id'],
                            'fighter1_id': row['fighter1_id'],
                            'fighter2_id': row['fighter2_id'],
                            'fighter1_name': row['fighter1_name'],
                            'fighter2_name': row['fighter2_name'],
                            'algopick_prediction': row['algopick_prediction'],
                            'algopick_probability': row['algopick_probability'],
                            'date': row['date']
                        }
                    else:
                        return {
                            'fight_id': row[0],
                            'fighter1_id': row[1],
                            'fighter2_id': row[2],
                            'fighter1_name': row[3],
                            'fighter2_name': row[4],
                            'algopick_prediction': row[5],
                            'algopick_probability': row[6],
                            'date': row[7]
                        }
        
        return None
    
    def get_fighter_id_from_name(self, api_name: str, fight_data: Dict) -> Optional[str]:
        """Map API fighter name to fighter_id from the fight data"""
        # Check if API name matches fighter1_name
        if fight_data['fighter1_name'] and api_name.lower() in fight_data['fighter1_name'].lower():
            return fight_data['fighter1_id']
        # Check if API name matches fighter2_name
        elif fight_data['fighter2_name'] and api_name.lower() in fight_data['fighter2_name'].lower():
            return fight_data['fighter2_id']
        
        # Try reverse - check if DB name is in API name
        if fight_data['fighter1_name'] and fight_data['fighter1_name'].lower() in api_name.lower():
            return fight_data['fighter1_id']
        elif fight_data['fighter2_name'] and fight_data['fighter2_name'].lower() in api_name.lower():
            return fight_data['fighter2_id']
        
        return None
    
    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def calculate_implied_probability(self, american_odds: int) -> float:
        """Calculate implied probability from American odds"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def calculate_ev(self, true_prob: float, odds: int) -> float:
        """Calculate Expected Value
        EV = (true_probability * decimal_odds) - 1
        Positive EV means profitable bet
        """
        decimal_odds = self.american_to_decimal(odds)
        return (true_prob * decimal_odds) - 1
    
    def get_fighter_probabilities(self, fight_data: Dict) -> Optional[Dict]:
        """Get model predictions for fighters from fight data
        Returns dict with fighter_id as key and probability as value
        """
        prediction = fight_data.get('algopick_prediction')
        probability = fight_data.get('algopick_probability')
        
        # If no prediction available
        if prediction is None or probability is None:
            return None
        
        # Convert to float if it's a Decimal
        probability = float(probability)
        
        fighter1_id = fight_data['fighter1_id']
        fighter2_id = fight_data['fighter2_id']
        
        # If prediction is 0, fighter1 has the stated probability
        # If prediction is 1, fighter2 has the stated probability
        if prediction == 0:
            fighter1_prob = probability / 100.0
            fighter2_prob = (100 - probability) / 100.0
        else:  # prediction == 1
            fighter2_prob = probability / 100.0
            fighter1_prob = (100 - probability) / 100.0
        
        return {
            fighter1_id: fighter1_prob,
            fighter2_id: fighter2_prob
        }
    
    def insert_complete_bookmaker_odds(self, fight_id: str, bookmaker: str, 
                                       fighters_data: List[Dict], 
                                       fight_data: Dict) -> None:
        """Insert or update complete bookmaker odds row with both fighters"""
        # Organize data by fighter position
        fighter1_data = None
        fighter2_data = None
        
        for fighter in fighters_data:
            if fighter['is_fighter1']:
                fighter1_data = fighter
            else:
                fighter2_data = fighter
        
        if not fighter1_data or not fighter2_data:
            print(f"      ⚠ Missing fighter data for complete insert")
            return
        
        # Calculate vigor and convert to percentage (e.g., 0.0462 -> 4.62)
        total_prob = fighter1_data['implied_prob'] + fighter2_data['implied_prob']
        vigor = (total_prob - 1.0) * 100
        
        # Convert implied probabilities to percentages (e.g., 0.6923 -> 69.23)
        fighter1_odds_percent = fighter1_data['implied_prob'] * 100
        fighter2_odds_percent = fighter2_data['implied_prob'] * 100
        
        # Convert EV to percentage (e.g., -0.0542 -> -5.42)
        ev_percent = (fighter1_data['ev'] * 100) if fighter1_data['ev'] is not None else None
        
        # Check if record exists
        check_query = """
            SELECT fight_id FROM ufc.bookmaker_odds
            WHERE fight_id = %s AND bookmaker = %s
        """
        
        result = fetch_query(self.conn, check_query, (fight_id, bookmaker))
        
        cursor = self.conn.cursor()
        
        if result and len(result) > 0:
            # Update existing record
            update_query = """
                UPDATE ufc.bookmaker_odds
                SET fighter1_id = %s,
                    fighter1_odds = %s,
                    fighter1_odds_percent = %s,
                    fighter2_id = %s,
                    fighter2_odds = %s,
                    fighter2_odds_percent = %s,
                    ev = %s,
                    vigor = %s
                WHERE fight_id = %s AND bookmaker = %s
            """
            
            cursor.execute(update_query, (
                fighter1_data['fighter_id'],
                fighter1_data['odds'],
                fighter1_odds_percent,
                fighter2_data['fighter_id'],
                fighter2_data['odds'],
                fighter2_odds_percent,
                ev_percent,
                vigor,
                fight_id,
                bookmaker
            ))
        else:
            # Insert new record
            insert_query = """
                INSERT INTO ufc.bookmaker_odds
                (fight_id, bookmaker, fighter1_id, fighter1_odds, fighter1_odds_percent,
                 fighter2_id, fighter2_odds, fighter2_odds_percent, ev, vigor)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                fight_id,
                bookmaker,
                fighter1_data['fighter_id'],
                fighter1_data['odds'],
                fighter1_odds_percent,
                fighter2_data['fighter_id'],
                fighter2_data['odds'],
                fighter2_odds_percent,
                ev_percent,
                vigor
            ))
        
        self.conn.commit()
        cursor.close()
    
    def process_odds(self, target_date: Optional[str] = None) -> None:
        """Main processing function"""
        data = self.load_odds_data()
        
        processed = 0
        skipped = 0
        errors = 0
        
        for event in data:
            try:
                home = event.get("home_team")
                away = event.get("away_team")
                event_time = event.get('commence_time')
                bookmakers = event.get('bookmakers', [])
                
                # Extract date from ISO format
                event_date = event_time.split('T')[0]
                
                # # Filter by date if specified
                # if target_date and not event_time.startswith(target_date):
                #     skipped += 1
                #     continue
                
                if not bookmakers:
                    skipped += 1
                    continue
                
                print(f"\n{'='*70}")
                print(f"API Fight: {home} vs {away}")
                print(f"Event Time: {event_time}")
                print(f"Event Date: {event_date}")
                
                # Find fight in database
                fight_data = self.find_fight_by_names(home, away, event_date)
                
                if not fight_data:
                    print(f"  ✗ No match found in database")
                    errors += 1
                    continue
                
                print(f"  ✓ MATCH FOUND!")
                print(f"    Fight ID: {fight_data['fight_id']}")
                print(f"    DB Fighter 1: {fight_data['fighter1_name']} (ID: {fight_data['fighter1_id']})")
                print(f"    DB Fighter 2: {fight_data['fighter2_name']} (ID: {fight_data['fighter2_id']})")
                print(f"    Fight Date: {fight_data['date']}")
                
                # Get model probabilities
                fighter_probs = self.get_fighter_probabilities(fight_data)
                
                if fighter_probs:
                    print(f"  ✓ Model Predictions:")
                    pred_text = "Fighter 1" if fight_data['algopick_prediction'] == 0 else "Fighter 2"
                    print(f"    Predicted Winner: {pred_text}")
                    print(f"    {fight_data['fighter1_name']}: {fighter_probs[fight_data['fighter1_id']]*100:.2f}%")
                    print(f"    {fight_data['fighter2_name']}: {fighter_probs[fight_data['fighter2_id']]*100:.2f}%")
                else:
                    print(f"  ⚠ No model predictions available")
                
                # Process each bookmaker's odds
                print(f"\n  Bookmaker Odds:")
                for bookmaker in bookmakers:
                    bm_name = bookmaker['key']
                    bm_title = bookmaker['title']
                    markets = bookmaker.get('markets', [])
                    
                    for market in markets:
                        if market['key'] != 'h2h':
                            continue
                        
                        outcomes = market.get('outcomes', [])
                        print(f"    → {bm_title} ({bm_name}):")
                        
                        # Collect both fighters' data before inserting
                        fighters_data = []
                        for outcome in outcomes:
                            api_fighter_name = outcome['name']
                            odds = outcome['price']
                            
                            # Map API name to fighter_id
                            fighter_id = self.get_fighter_id_from_name(api_fighter_name, fight_data)
                            
                            if not fighter_id:
                                print(f"      ⚠ Could not map {api_fighter_name} to a fighter_id")
                                continue
                            
                            # Determine if this is fighter1 or fighter2
                            is_fighter1 = (fighter_id == fight_data['fighter1_id'])
                            
                            # Calculate metrics
                            implied_prob = self.calculate_implied_probability(odds)
                            
                            # Calculate EV if available
                            ev_val = None
                            if fighter_probs and fighter_id in fighter_probs:
                                ev_val = self.calculate_ev(fighter_probs[fighter_id], odds)
                                ev_percent = ev_val * 100
                                ev_indicator = "🟢" if ev_val > 0 else "🔴"
                                print(f"      {api_fighter_name}: {odds:+d} (Implied: {implied_prob*100:.2f}%, EV: {ev_indicator} {ev_percent:+.2f}%)")
                            else:
                                print(f"      {api_fighter_name}: {odds:+d} (Implied: {implied_prob*100:.2f}%)")
                            
                            fighters_data.append({
                                'fighter_id': fighter_id,
                                'odds': odds,
                                'implied_prob': implied_prob,
                                'ev': ev_val,
                                'is_fighter1': is_fighter1
                            })
                        
                        # Insert/update with both fighters' data at once
                        if len(fighters_data) == 2:
                            self.insert_complete_bookmaker_odds(
                                fight_data['fight_id'],
                                bm_name,
                                fighters_data,
                                fight_data
                            )
                        else:
                            print(f"      ⚠ Incomplete data: only {len(fighters_data)} fighters found")
                
                processed += 1
                
            except Exception as e:
                print(f"  ✗ Error processing event: {str(e)}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                errors += 1
                continue
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Processed: {processed}")
        print(f"  Skipped: {skipped}")
        print(f"  Errors: {errors}")
        print(f"{'='*70}\n")

    def close(self):
        """Close database connection"""
        # Connection managed externally, so just pass
        pass


def run():
    conn = create_connection()
    processor = OddsProcessor(conn, "ufc_odds.json")
    
    try:
        # Process all odds for a specific date
        processor.process_odds(target_date="2025-10-24")
        
        # Or process all odds regardless of date
        # processor.process_odds()
        
    finally:
        processor.close()
        conn.close()