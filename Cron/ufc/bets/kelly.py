from .utils import create_connection, fetch_query
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
import mysql.connector
from datetime import datetime

# Set decimal precision for financial calculations
getcontext().prec = 10

class KellyAnalyticsBuilder:
    def __init__(self):
        self.conn = create_connection()
        
    def create_analytics_tables(self):
        """Create the analytics tables if they don't exist"""
        
        # Main bet analytics table
        bet_analytics_sql = """
        CREATE TABLE IF NOT EXISTS bet_analytics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            fight_id VARCHAR(255),
            event_id VARCHAR(255),
            bet_sequence INT,
            strategy_name VARCHAR(100),
            
            -- Bet details
            bet_size DECIMAL(10,2),
            win_probability DECIMAL(5,4),
            decimal_odds DECIMAL(6,3),
            kelly_fraction DECIMAL(6,4),
            expected_value DECIMAL(10,2),
            
            -- Bankroll tracking
            bankroll_before DECIMAL(10,2),
            bankroll_after DECIMAL(10,2),
            cumulative_profit DECIMAL(10,2),
            
            -- Risk metrics
            bet_outcome ENUM('won', 'lost', 'pending', 'void'),
            actual_profit DECIMAL(10,2),
            running_roi DECIMAL(8,4),
            max_drawdown DECIMAL(10,2),
            
            -- Streak tracking
            current_win_streak INT DEFAULT 0,
            current_loss_streak INT DEFAULT 0,
            
            bet_date DATETIME,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX(strategy_name, bet_sequence),
            INDEX(fight_id),
            INDEX(event_id),
            INDEX(bet_date)
        );
        """
        
        # Risk metrics summary table
        risk_metrics_sql = """
        CREATE TABLE IF NOT EXISTS risk_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            strategy_name VARCHAR(100),
            calculation_date DATE,
            
            -- Performance metrics
            total_bets INT,
            win_rate DECIMAL(5,4),
            total_profit DECIMAL(10,2),
            roi DECIMAL(8,4),
            
            -- Risk metrics
            max_drawdown DECIMAL(10,2),
            current_drawdown DECIMAL(10,2),
            sharpe_ratio DECIMAL(6,4),
            volatility DECIMAL(6,4),
            
            -- Kelly-specific
            avg_kelly_fraction DECIMAL(6,4),
            kelly_utilization DECIMAL(5,4),
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY(strategy_name, calculation_date)
        );
        """
        
        cursor = self.conn.cursor()
        cursor.execute(bet_analytics_sql)
        cursor.execute(risk_metrics_sql)
        self.conn.commit()
        cursor.close()
        print("Analytics tables created successfully!")
    
    def get_base_bet_data(self, event_id=None):
        """Fetch and prepare base betting data"""
        query = """
        SELECT 
            fight_id,
            event_id,
            fighter1_name,
            fighter2_name,
            fighter1_odds,
            fighter2_odds,
            fighter1_pred,
            fighter2_pred,
            fighter_bet_on,
            bet_outcome,
            potential_profit,
            stake,
            bet_date
        FROM bets 
        WHERE bet_outcome NOT IN ('pending', 'void')
        """
        
        if event_id:
            query += f" AND event_id = '{event_id}'"
            
        query += " ORDER BY bet_date, fight_id"
        
        return fetch_query(self.conn, query)
    
    def calculate_kelly_metrics(self, row):
        """Calculate Kelly fraction and related metrics for a single bet"""
        
        # Determine which fighter we bet on and get win probability
        if row['fighter_bet_on'] == 0:
            win_prob = float(row['fighter1_pred']) / 100.0
            american_odds = float(row['fighter1_odds'])
        else:
            win_prob = float(row['fighter2_pred']) / 100.0
            american_odds = float(row['fighter2_odds'])
        
        # Convert American odds to decimal odds
        if american_odds > 0:
            decimal_odds = (american_odds / 100.0) + 1
        else:
            decimal_odds = (100.0 / abs(american_odds)) + 1
        
        # Calculate Kelly fraction: f = (bp - q) / b
        # where b = decimal_odds - 1, p = win_prob, q = 1 - win_prob
        b = decimal_odds - 1
        p = win_prob
        q = 1 - win_prob
        
        kelly_fraction = max(0, (b * p - q) / b) if b > 0 else 0
        
        # Expected value = (win_prob * payout) - (loss_prob * bet_amount)
        expected_value = (win_prob * b) - (1 - win_prob)
        
        return {
            'win_probability': win_prob,
            'decimal_odds': decimal_odds,
            'kelly_fraction': kelly_fraction,
            'expected_value': expected_value
        }
    
    def simulate_strategy(self, df, strategy_name, kelly_multiplier, max_bet_pct=0.10, starting_bankroll=1000):
        """Simulate a betting strategy with sequential bankroll tracking"""
        
        results = []
        current_bankroll = starting_bankroll
        bet_sequence = 0
        cumulative_profit = 0
        peak_bankroll = starting_bankroll
        max_drawdown = 0
        win_streak = 0
        loss_streak = 0
        
        for idx, row in df.iterrows():
            bet_sequence += 1
            
            # Calculate Kelly metrics
            kelly_metrics = self.calculate_kelly_metrics(row)
            
            # Determine bet size
            full_kelly = kelly_metrics['kelly_fraction']
            adjusted_kelly = full_kelly * kelly_multiplier
            bet_fraction = min(adjusted_kelly, max_bet_pct)
            bet_size = current_bankroll * bet_fraction
            
            # Skip if bet size too small or no edge
            if bet_size < 1:
                continue
            
            bankroll_before = current_bankroll
            
            # Calculate actual profit/loss
            if row['bet_outcome'] == 'won':
                actual_profit = bet_size * (kelly_metrics['decimal_odds'] - 1)
                win_streak += 1
                loss_streak = 0
            elif row['bet_outcome'] == 'lost':
                actual_profit = -bet_size
                loss_streak += 1
                win_streak = 0
            else:
                continue  # Skip pending/void
            
            # Update bankroll
            current_bankroll += actual_profit
            cumulative_profit += actual_profit
            
            # Track drawdown
            if current_bankroll > peak_bankroll:
                peak_bankroll = current_bankroll
            
            current_drawdown = peak_bankroll - current_bankroll
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Calculate running ROI
            total_staked = sum([r['bet_size'] for r in results]) + bet_size
            running_roi = (cumulative_profit / total_staked) if total_staked > 0 else 0
            
            # Store result
            result = {
                'fight_id': row['fight_id'],
                'event_id': row['event_id'],
                'bet_sequence': bet_sequence,
                'strategy_name': strategy_name,
                'bet_size': round(bet_size, 2),
                'win_probability': round(kelly_metrics['win_probability'], 4),
                'decimal_odds': round(kelly_metrics['decimal_odds'], 3),
                'kelly_fraction': round(full_kelly, 4),
                'expected_value': round(kelly_metrics['expected_value'] * bet_size, 2),
                'bankroll_before': round(bankroll_before, 2),
                'bankroll_after': round(current_bankroll, 2),
                'cumulative_profit': round(cumulative_profit, 2),
                'bet_outcome': row['bet_outcome'],
                'actual_profit': round(actual_profit, 2),
                'running_roi': round(running_roi, 4),
                'max_drawdown': round(max_drawdown, 2),
                'current_win_streak': win_streak,
                'current_loss_streak': loss_streak,
                'bet_date': row['bet_date']
            }
            
            results.append(result)
        
        return results
    
    def simulate_flat_betting(self, df, bet_amount=50, starting_bankroll=1000):
        """Simulate flat betting strategy"""
        
        results = []
        current_bankroll = starting_bankroll
        bet_sequence = 0
        cumulative_profit = 0
        peak_bankroll = starting_bankroll
        max_drawdown = 0
        win_streak = 0
        loss_streak = 0
        
        for idx, row in df.iterrows():
            bet_sequence += 1
            bankroll_before = current_bankroll
            
            # Flat bet amount
            bet_size = bet_amount
            
            # Calculate actual profit/loss from your existing data
            if row['bet_outcome'] == 'won':
                actual_profit = float(row['potential_profit'])
                win_streak += 1
                loss_streak = 0
            elif row['bet_outcome'] == 'lost':
                actual_profit = -float(row['stake'])
                loss_streak += 1
                win_streak = 0
            else:
                continue
            
            # Update bankroll (for tracking purposes)
            current_bankroll += actual_profit
            cumulative_profit += actual_profit
            
            # Track drawdown
            if current_bankroll > peak_bankroll:
                peak_bankroll = current_bankroll
            
            current_drawdown = peak_bankroll - current_bankroll
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Running ROI
            total_staked = bet_sequence * bet_amount
            running_roi = cumulative_profit / total_staked
            
            # Get Kelly metrics for comparison
            kelly_metrics = self.calculate_kelly_metrics(row)
            
            result = {
                'fight_id': row['fight_id'],
                'event_id': row['event_id'],
                'bet_sequence': bet_sequence,
                'strategy_name': 'Flat_50',
                'bet_size': bet_size,
                'win_probability': round(kelly_metrics['win_probability'], 4),
                'decimal_odds': round(kelly_metrics['decimal_odds'], 3),
                'kelly_fraction': round(kelly_metrics['kelly_fraction'], 4),
                'expected_value': round(kelly_metrics['expected_value'] * bet_size, 2),
                'bankroll_before': round(bankroll_before, 2),
                'bankroll_after': round(current_bankroll, 2),
                'cumulative_profit': round(cumulative_profit, 2),
                'bet_outcome': row['bet_outcome'],
                'actual_profit': round(actual_profit, 2),
                'running_roi': round(running_roi, 4),
                'max_drawdown': round(max_drawdown, 2),
                'current_win_streak': win_streak,
                'current_loss_streak': loss_streak,
                'bet_date': row['bet_date']
            }
            
            results.append(result)
        
        return results
    
    def populate_bet_analytics(self, event_id=None, clear_existing=True):
        """Populate the bet_analytics table with all strategies
        
        Args:
            event_id (str, optional): Only process bets from this specific event
            clear_existing (bool): Whether to clear existing data before inserting
        """
        
        print("Fetching base bet data...")
        if event_id:
            print(f"Filtering for event: {event_id}")
            
        df = pd.DataFrame(self.get_base_bet_data(event_id))
        
        if df.empty:
            print("No bet data found!")
            return
        
        print(f"Processing {len(df)} bets...")
        
        # Clear existing data only if requested
        if clear_existing:
            cursor = self.conn.cursor()
            if event_id:
                # Only clear data for this specific event
                cursor.execute("DELETE FROM bet_analytics WHERE fight_id IN (SELECT fight_id FROM bets WHERE event_id = %s)", (event_id,))
            else:
                # Clear all data
                cursor.execute("DELETE FROM bet_analytics")
            self.conn.commit()
            cursor.close()
        
        # Define strategies to simulate
        strategies = [
            ('Flat_50', 0, 0.10),  # (name, kelly_multiplier, max_bet_pct)
            ('Kelly_4pct', 0.04, 0.10),
            ('Kelly_5pct', 0.05, 0.10),
            ('Kelly_6pct', 0.06, 0.10),
            ('Kelly_7pct', 0.07, 0.10),
            ('Kelly_8pct', 0.08, 0.10),
            ('Kelly_9pct', 0.09, 0.10),
            ('Kelly_10pct', 0.10, 0.10),
            ('Kelly_11pct', 0.11, 0.10)
        ]
        
        all_results = []
        
        for strategy_name, kelly_mult, max_bet in strategies:
            print(f"Simulating {strategy_name}...")
            
            if strategy_name == 'Flat_50':
                results = self.simulate_flat_betting(df)
            else:
                results = self.simulate_strategy(df, strategy_name, kelly_mult, max_bet)
            
            all_results.extend(results)
        
        # Insert all results
        print("Inserting results into database...")
        insert_sql = """
        INSERT INTO bet_analytics (
            fight_id, event_id, bet_sequence, strategy_name, bet_size, win_probability,
            decimal_odds, kelly_fraction, expected_value, bankroll_before,
            bankroll_after, cumulative_profit, bet_outcome, actual_profit,
            running_roi, max_drawdown, current_win_streak, current_loss_streak,
            bet_date
        ) VALUES (
            %(fight_id)s, %(event_id)s, %(bet_sequence)s, %(strategy_name)s, %(bet_size)s,
            %(win_probability)s, %(decimal_odds)s, %(kelly_fraction)s,
            %(expected_value)s, %(bankroll_before)s, %(bankroll_after)s,
            %(cumulative_profit)s, %(bet_outcome)s, %(actual_profit)s,
            %(running_roi)s, %(max_drawdown)s, %(current_win_streak)s,
            %(current_loss_streak)s, %(bet_date)s
        )
        """
        
        cursor.executemany(insert_sql, all_results)
        self.conn.commit()
        cursor.close()
        
        print(f"Inserted {len(all_results)} analytics records!")
    
    def calculate_risk_metrics(self):
        """Calculate and populate risk metrics summary table"""
        
        print("Calculating risk metrics...")
        
        # Clear existing risk metrics
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM risk_metrics")
        
        # Get strategy performance data
        query = """
        SELECT 
            strategy_name,
            COUNT(*) as total_bets,
            AVG(CASE WHEN bet_outcome = 'won' THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(actual_profit) as total_profit,
            SUM(bet_size) as total_staked,
            MAX(max_drawdown) as max_drawdown
        FROM bet_analytics
        WHERE bet_outcome IN ('won', 'lost')
        GROUP BY strategy_name
        """
        
        strategies_data = fetch_query(self.conn, query)
        
        risk_metrics = []
        
        for strategy in strategies_data:
            strategy_name = strategy['strategy_name']
            
            # Calculate additional metrics
            roi = float(strategy['total_profit']) / float(strategy['total_staked']) if strategy['total_staked'] > 0 else 0
            
            # Get return series for Sharpe ratio calculation
            returns_query = f"""
            SELECT (actual_profit / bet_size) as return_rate
            FROM bet_analytics 
            WHERE strategy_name = '{strategy_name}' AND bet_outcome IN ('won', 'lost')
            """
            returns_data = fetch_query(self.conn, returns_query)
            returns = [float(r['return_rate']) for r in returns_data]
            
            # Calculate Sharpe ratio and volatility
            if len(returns) > 1:
                avg_return = float(np.mean(returns))
                volatility = float(np.std(returns))
                sharpe_ratio = (avg_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
            
            # Calculate actual Kelly fraction used based on strategy
            if strategy_name == 'Flat_50':
                avg_kelly_fraction = 0.0  # Flat betting doesn't use Kelly
            else:
                # Extract multiplier from strategy name and calculate actual Kelly used
                kelly_query = f"""
                SELECT AVG(kelly_fraction) as base_kelly_fraction
                FROM bet_analytics 
                WHERE strategy_name = '{strategy_name}' AND bet_outcome IN ('won', 'lost')
                """
                kelly_data = fetch_query(self.conn, kelly_query)
                base_kelly = float(kelly_data[0]['base_kelly_fraction']) if kelly_data and kelly_data[0]['base_kelly_fraction'] else 0
                
                # Extract the multiplier from strategy name
                if 'Kelly_4pct' in strategy_name:
                    multiplier = 0.04
                elif 'Kelly_5pct' in strategy_name:
                    multiplier = 0.05
                elif 'Kelly_6pct' in strategy_name:
                    multiplier = 0.06
                elif 'Kelly_7pct' in strategy_name:
                    multiplier = 0.07
                elif 'Kelly_8pct' in strategy_name:
                    multiplier = 0.08
                elif 'Kelly_9pct' in strategy_name:
                    multiplier = 0.09
                elif 'Kelly_10pct' in strategy_name:
                    multiplier = 0.10
                elif 'Kelly_11pct' in strategy_name:
                    multiplier = 0.11
                else:
                    multiplier = 0.05  # Default fallback
                
                # Calculate actual Kelly fraction used (base Kelly * multiplier)
                avg_kelly_fraction = base_kelly * multiplier
            
            # Kelly utilization (% of opportunities where we bet)
            # Count total opportunities from original bets table, not just placed bets
            total_opportunities_query = """
            SELECT COUNT(*) as count 
            FROM bets 
            WHERE bet_outcome NOT IN ('pending', 'void')
            """
            total_opportunities_result = fetch_query(self.conn, total_opportunities_query)
            total_opportunities = total_opportunities_result[0]['count'] if total_opportunities_result else 0
            kelly_utilization = float(strategy['total_bets']) / float(total_opportunities) if total_opportunities > 0 else 0
            
            # Current drawdown (from latest peak)
            current_drawdown_query = f"""
            WITH running_peak AS (
                SELECT bankroll_after,
                    MAX(bankroll_after) OVER (ORDER BY bet_sequence ROWS UNBOUNDED PRECEDING) as peak
                FROM bet_analytics 
                WHERE strategy_name = '{strategy_name}'
                ORDER BY bet_sequence DESC 
                LIMIT 1
            )
            SELECT peak - bankroll_after as current_drawdown FROM running_peak
            """
            current_drawdown_result = fetch_query(self.conn, current_drawdown_query)
            current_drawdown = float(current_drawdown_result[0]['current_drawdown']) if current_drawdown_result else 0.0
            
            risk_metric = {
                'strategy_name': strategy_name,
                'calculation_date': datetime.now().date(),
                'total_bets': int(strategy['total_bets']),
                'win_rate': round(float(strategy['win_rate']), 4),
                'total_profit': round(float(strategy['total_profit']), 2),
                'roi': round(roi, 4),
                'max_drawdown': round(float(strategy['max_drawdown']), 2),
                'current_drawdown': round(current_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'volatility': round(volatility, 4),
                'avg_kelly_fraction': round(avg_kelly_fraction, 4),
                'kelly_utilization': round(kelly_utilization, 4)
            }
            
            risk_metrics.append(risk_metric)
        
        # Insert risk metrics
        insert_sql = """
        INSERT INTO risk_metrics (
            strategy_name, calculation_date, total_bets, win_rate, total_profit,
            roi, max_drawdown, current_drawdown, sharpe_ratio, volatility,
            avg_kelly_fraction, kelly_utilization
        ) VALUES (
            %(strategy_name)s, %(calculation_date)s, %(total_bets)s, %(win_rate)s,
            %(total_profit)s, %(roi)s, %(max_drawdown)s, %(current_drawdown)s,
            %(sharpe_ratio)s, %(volatility)s, %(avg_kelly_fraction)s, %(kelly_utilization)s
        )
        """
        
        cursor.executemany(insert_sql, risk_metrics)
        self.conn.commit()
        cursor.close()
        
        print(f"Calculated risk metrics for {len(risk_metrics)} strategies!")
    
    def run_full_analysis(self, create_tables=False, event_id=None, clear_existing=True):
        """Run the complete analytics pipeline
        
        Args:
            create_tables (bool): Whether to create tables first
            event_id (str, optional): Only process bets from this specific event
            clear_existing (bool): Whether to clear existing data before inserting
        """
        print("Starting Kelly Analytics Builder...")
        
        try:
            if create_tables:
                self.create_analytics_tables()
            self.populate_bet_analytics(event_id=event_id, clear_existing=False)
            self.calculate_risk_metrics()
            
            print("\n" + "="*50)
            print("ANALYTICS COMPLETE!")
            print("="*50)
            print("Tables populated:")
            print("- bet_analytics: Sequential bet tracking with Kelly sizing")
            print("- risk_metrics: Strategy performance summaries")
            if event_id:
                print(f"- Filtered for event: {event_id}")
            print("\nReady for UI dashboard integration!")
            
        except Exception as e:
            print(f"Error in analytics pipeline: {str(e)}")
            raise
        finally:
            if self.conn:
                self.conn.close()
    
    def get_bankroll_chart_data(self, strategies=None):
        """Get data for x/y bankroll growth chart"""
        if strategies is None:
            strategies = ['Flat_50', 'Kelly_5pct', 'Kelly_8pct', 'Kelly_10pct']
        
        strategy_list = "'" + "','".join(strategies) + "'"
        
        query = f"""
        SELECT strategy_name, bet_sequence, bankroll_after, bet_date
        FROM bet_analytics 
        WHERE strategy_name IN ({strategy_list})
        ORDER BY strategy_name, bet_sequence
        """
        
        return fetch_query(self.conn, query)

# Usage
def run(event_id=None, clear_existing=False):
    """Run Kelly analytics for all bets or specific event
    
    Args:
        event_id (str, optional): Only process bets from this specific event
        clear_existing (bool): Whether to clear existing data before inserting
    """
    builder = KellyAnalyticsBuilder()
    builder.run_full_analysis(create_tables=False, event_id=event_id, clear_existing=clear_existing)
    
    if event_id:
        print(f"✅ Analytics complete for event: {event_id}")
    else:
        print("✅ All done! Your analytics tables are ready for your UI.")