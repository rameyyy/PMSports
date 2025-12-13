#!/usr/bin/env python3

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import sys
import argparse
from dotenv import load_dotenv

# Add current directory to path for imports
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

# Load environment variables from .env file
load_dotenv(os.path.join(ncaamb_dir, '.env'))

from scrapes import sqlconn


def get_yesterdays_results(target_date=None):
    """
    Query database for yesterday's betting results from my_bankroll table.

    Args:
        target_date: Date string in YYYY-MM-DD format. If provided, gets results for the day before this date.
                    If None, uses yesterday (day before today).

    Returns:
        dict: Dictionary with yesterday's results or None if no data found
    """
    from datetime import timedelta

    # Calculate yesterday's date
    if target_date:
        current_date = datetime.strptime(target_date, '%Y-%m-%d')
    else:
        current_date = datetime.now()

    yesterday = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')

    conn = sqlconn.create_connection()
    if not conn:
        print("[-] Could not connect to database for yesterday's results")
        return None

    try:
        query = """
            SELECT
                ml_bets,
                ml_wins,
                ml_roi,
                ml_net_profit_loss,
                ou_bets,
                ou_wins,
                ou_roi,
                ou_net_profit_loss,
                net_profit_loss
            FROM my_bankroll
            WHERE date = %s
        """

        results = sqlconn.fetch(conn, query, (yesterday,))
        conn.close()

        if results and len(results) > 0:
            return results[0]
        else:
            return None

    except Exception as e:
        print(f"[-] Error querying yesterday's results: {e}")
        if conn:
            conn.close()
        return None


def get_todays_bets(target_date=None):
    """
    Query database for moneyline and overunder bets for a specific date.

    Args:
        target_date: Date string in YYYY-MM-DD format. If None, uses today's date.

    Returns:
        tuple: (moneyline_bets, overunder_bets) as lists of dictionaries
    """
    if target_date:
        date_str = target_date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')

    conn = sqlconn.create_connection()
    if not conn:
        print("[-] Could not connect to database")
        return [], []

    try:
        # Query moneyline bets for today where bet_on = 1
        ml_query = """
            SELECT
                game_id,
                team_1,
                team_2,
                team_predicted_to_win,
                ensemble_prob_team_1,
                ensemble_prob_team_2,
                my_best_ev_team_1,
                my_best_ev_team_2,
                my_best_book_team_1,
                my_best_book_odds_team_1,
                my_best_book_team_2,
                my_best_book_odds_team_2,
                best_book_odds_team_1,
                best_book_odds_team_2,
                bet_rule
            FROM moneyline
            WHERE game_date = %s AND bet_on = 1
            ORDER BY game_id
        """

        ml_bets = sqlconn.fetch(conn, ml_query, (date_str,))

        # Query overunder bets for specified date where bet_on = 1
        ou_query = """
            SELECT
                game_id,
                team_1,
                team_2,
                over_point,
                ensemble_pred,
                ensemble_confidence,
                good_bets_confidence,
                difference,
                bet_on_side,
                my_best_book_over,
                my_best_book_odds_over,
                my_best_book_under,
                my_best_book_odds_under,
                bet_rule
            FROM overunder
            WHERE game_date = %s AND bet_on = 1
            ORDER BY game_id
        """

        ou_bets = sqlconn.fetch(conn, ou_query, (date_str,))

        conn.close()

        return ml_bets, ou_bets

    except Exception as e:
        print(f"[-] Error querying database: {e}")
        if conn:
            conn.close()
        return [], []


def format_american_odds(odds):
    """Format odds as American odds with + or - sign"""
    if odds is None:
        return "N/A"
    odds = int(odds)
    return f"+{odds}" if odds > 0 else str(odds)


def format_email_body(ml_bets, ou_bets, yesterday_results=None):
    """
    Format the email body with today's bets and yesterday's results.

    Args:
        ml_bets: List of moneyline bet dictionaries
        ou_bets: List of overunder bet dictionaries
        yesterday_results: Dictionary with yesterday's results from my_bankroll table

    Returns:
        str: Formatted email body
    """
    body = ""

    # Yesterday's results section
    if yesterday_results:
        ml_wins = yesterday_results.get('ml_wins', 0) or 0
        ml_total = yesterday_results.get('ml_bets', 0) or 0
        ml_roi = yesterday_results.get('ml_roi', 0)
        ml_roi = float(ml_roi) if ml_roi is not None else 0.0
        ml_profit = yesterday_results.get('ml_net_profit_loss', 0)
        ml_profit = float(ml_profit) if ml_profit is not None else 0.0

        ou_wins = yesterday_results.get('ou_wins', 0) or 0
        ou_total = yesterday_results.get('ou_bets', 0) or 0
        ou_roi = yesterday_results.get('ou_roi', 0)
        ou_roi = float(ou_roi) if ou_roi is not None else 0.0
        ou_profit = yesterday_results.get('ou_net_profit_loss', 0)
        ou_profit = float(ou_profit) if ou_profit is not None else 0.0

        total_profit = yesterday_results.get('net_profit_loss', 0)
        total_profit = float(total_profit) if total_profit is not None else 0.0

        body += "YESTERDAY'S RESULTS\n\n"
        body += f"Moneyline: {ml_wins}/{ml_total} wins | ROI: {ml_roi:.2f}% | Profit: ${ml_profit:.2f}\n"
        body += f"Over/Under: {ou_wins}/{ou_total} wins | ROI: {ou_roi:.2f}% | Profit: ${ou_profit:.2f}\n"
        body += f"Total Net Profit: ${total_profit:.2f}\n\n"

    # Today's bets header
    body += "TODAY'S BETS\n\n"

    # Today's bets summary
    total_bets = len(ml_bets) + len(ou_bets)
    body += f"Total Bets: {total_bets}\n"
    body += f"Moneyline Bets: {len(ml_bets)}\n"
    body += f"Totals Bets: {len(ou_bets)}\n\n"

    # Moneyline bets
    if ml_bets:
        body += "MONEYLINE BETS\n\n"
        for bet in ml_bets:
            game = f"{bet['team_1']} vs {bet['team_2']}"
            predicted_winner = bet['team_predicted_to_win']

            # Determine which team is predicted to win and get their odds and win probability
            if predicted_winner == bet['team_1']:
                best_odds = format_american_odds(bet['best_book_odds_team_1'])
                win_prob = float(bet['ensemble_prob_team_1']) * 100 if bet['ensemble_prob_team_1'] else 0.0
            else:
                best_odds = format_american_odds(bet['best_book_odds_team_2'])
                win_prob = float(bet['ensemble_prob_team_2']) * 100 if bet['ensemble_prob_team_2'] else 0.0

            body += f"Game: {game}\n"
            body += f"BET: {predicted_winner} at {best_odds} (Win Prob: {win_prob:.2f}%)\n\n"

    # Over/Under bets
    if ou_bets:
        if ml_bets:
            body += "\n"
        body += "OVER/UNDER BETS\n\n"
        for bet in ou_bets:
            game = f"{bet['team_1']} vs {bet['team_2']}"
            bet_side = bet['bet_on_side']
            over_point = float(bet['over_point']) if bet['over_point'] else 0.0

            body += f"Game: {game}\n"
            body += f"BET: {bet_side} {over_point}\n\n"

    if not ml_bets and not ou_bets:
        body = "No bets for today.\n"

    return body


def send_daily_email(target_date=None):
    """
    Send a daily email to yourself via Gmail with bets for a specific date.

    Args:
        target_date: Date string in YYYY-MM-DD format. If None, uses today's date.
    """
    # Your Gmail credentials
    email = "clayrameyy@gmail.com"
    password = os.getenv('GMAIL_APP_CODE')

    if not password:
        print("[-] GMAIL_APP_CODE not found in .env file")
        return False

    # Determine the date to use
    if target_date:
        date_str = target_date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')

    # Get bets from database for the specified date
    print(f"Querying bets for {date_str} from database...")
    ml_bets, ou_bets = get_todays_bets(target_date=target_date)

    print(f"Found {len(ml_bets)} moneyline bets and {len(ou_bets)} over/under bets")

    # Get yesterday's results
    print(f"Querying yesterday's results from database...")
    yesterday_results = get_yesterdays_results(target_date=target_date)

    if yesterday_results:
        print(f"Found yesterday's results")
    else:
        print(f"No results found for yesterday")

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = email
    msg['Subject'] = f"AlgoPick Bet {date_str}"

    # Format email body with betting data and yesterday's results
    body = format_email_body(ml_bets, ou_bets, yesterday_results=yesterday_results)
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to Gmail's SMTP server
        print("Connecting to Gmail...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # Login
        print("Authenticating...")
        server.login(email, password)

        # Send the email
        print("Sending email...")
        server.send_message(msg)
        server.quit()

        print(f"Email sent successfully to {email} at {datetime.now().strftime('%I:%M %p')}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("Authentication failed!")
        print("Make sure you're using the correct Gmail App Password")
        print("Go to: Google Account -> Security -> 2-Step Verification -> App passwords")
        return False

    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def main():
    """Main function to send daily betting email"""
    parser = argparse.ArgumentParser(description='Send betting email for a specific date')
    parser.add_argument('--date', type=str, default=None, help='Date to get bets for in YYYY-MM-DD format (default: today)')
    args = parser.parse_args()

    # Validate date format if provided
    if args.date:
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"[-] Invalid date format: {args.date}. Expected YYYY-MM-DD")
            return

    send_daily_email(target_date=args.date)


if __name__ == "__main__":
    main()
