#!/usr/bin/env python3
"""
Main entry point for NCAAMB data scraping and processing
"""

from scrapes.gamehistory import scrape_game_history, scrape_all_teams

if __name__ == "__main__":
    # Scrape all teams from leaderboard and insert into MySQL
    print("Starting full scrape of all teams...\n")
    scrape_all_teams(year='2023', season=2023)
