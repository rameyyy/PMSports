#!/usr/bin/env python3
"""
Test script for building flat dataset with historical match data
"""
from models.build_flat_df import build_flat_df
import json

if __name__ == "__main__":
    print("="*100)
    print("Testing build_flat_df with games from 2019-11-15 (season 2020)")
    print("="*100 + "\n")

    try:
        # Build flat dataset for specific date
        data = build_flat_df(season=2020, target_date="2020-02-15")

        print("\n" + "="*100)
        print("RESULTS")
        print("="*100)
        print(f"\nTotal games: {len(data)}")

        print("\n" + "-"*100)
        print("First game details:")
        print("-"*100)
        first_game = data[0]
        print(f"  Game ID: {first_game['game_id']}")
        print(f"  Date: {first_game['date']}")
        print(f"  Season: {first_game['season']}")
        print(f"  Team 1: {first_game['team_1']}")
        print(f"  Team 1 Conference: {first_game['team_1_conference']}")
        print(f"  Team 1 Score: {first_game['team_1_score']}")
        print(f"  Team 1 Historical Games: {first_game['team_1_hist_count']}")
        print(f"  Team 2: {first_game['team_2']}")
        print(f"  Team 2 Conference: {first_game['team_2_conference']}")
        print(f"  Team 2 Score: {first_game['team_2_score']}")
        print(f"  Team 2 Historical Games: {first_game['team_2_hist_count']}")
        print(f"  Total Score: {first_game['total_score_outcome']}")
        print(f"  Team 1 Win/Loss: {first_game['team_1_winloss']} ({'Win' if first_game['team_1_winloss'] == 1 else 'Loss'})")

        # Leaderboard info
        print(f"\n  Team 1 Leaderboard Data (day before): {'Found' if first_game['team_1_leaderboard'] else 'Not Found'}")
        if first_game['team_1_leaderboard']:
            lb = first_game['team_1_leaderboard']
            print(f"    Rank: {lb.get('rank')}, W-L: {lb.get('wins')}-{lb.get('losses')}, Barthag: {lb.get('barthag')}")

        print(f"  Team 2 Leaderboard Data (day before): {'Found' if first_game['team_2_leaderboard'] else 'Not Found'}")
        if first_game['team_2_leaderboard']:
            lb = first_game['team_2_leaderboard']
            print(f"    Rank: {lb.get('rank')}, W-L: {lb.get('wins')}-{lb.get('losses')}, Barthag: {lb.get('barthag')}")

        if first_game['team_1_hist_count'] > 0:
            print("\n  Sample of Team 1's first historical game:")
            hist_game = first_game['team_1_match_hist'][0]
            print(f"    Game ID: {hist_game['game_id']}")
            print(f"    Date: {hist_game['date']}")
            print(f"    Teams: {hist_game['team_1']} vs {hist_game['team_2']}")
            print(f"    Score: {hist_game['team_1_score']} - {hist_game['team_2_score']}")
            print(f"    Winner: {hist_game['winner']}")
            print(f"    Team 1 Leaderboard (day before): {'Found' if hist_game.get('team_1_leaderboard') else 'Not Found'}")
            print(f"    Team 2 Leaderboard (day before): {'Found' if hist_game.get('team_2_leaderboard') else 'Not Found'}")
            print(f"    Team 1 Player Stats: {len(hist_game.get('team_1_player_stats', []))} players")
            print(f"    Team 2 Player Stats: {len(hist_game.get('team_2_player_stats', []))} players")

            if len(hist_game.get('team_1_player_stats', [])) > 0:
                player = hist_game['team_1_player_stats'][0]
                print(f"      Sample Player (Team 1): {player.get('player_name')} - {player.get('pts')} pts, {player.get('Min_per')} min")

        # Save to JSON for inspection
        output_file = "flat_df_sample.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print("\n" + "="*100)
        print(f"Sample saved to {output_file}")
        print("="*100)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
