export interface HomepageStats {
  date: string;
  todays_games_count: number;
  my_accuracy: number;
  vegas_accuracy: number;
  my_total_correct: number;
  vegas_total_correct: number;
  total_complete_matches: number;
  pick_of_day_acc: number;
  pick_of_day_correct: number;
  pick_of_day_total: number;
  pod_avg_odds: number;
  pod_roi: number;
  pod_td_matchup: string | null;
  pod_td_pick: string | null;
  pod_td_odds: number | null;
  pod_yd_matchup: string | null;
  pod_yd_pick: string | null;
  pod_yd_odds: number | null;
  pod_yd_outcome: string | null;
}

export async function fetchHomepageStats(): Promise<HomepageStats> {
  const res = await fetch('/api/ncaamb/homepage-stats');
  if (!res.ok) throw new Error('Failed to fetch homepage stats');
  return res.json();
}
