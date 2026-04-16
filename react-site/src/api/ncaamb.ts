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
  pod_units: number;
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

export interface Game {
  game_id: string;
  date: string;
  team_1: string;
  team_1_hna: 'H' | 'N' | 'A';
  team_2: string;
  team_1_rank: number | null;
  team_2_rank: number | null;
  team_1_conference: string | null;
  team_2_conference: string | null;
  team_1_prob_algopicks: number | null;
  team_2_prob_algopicks: number | null;
  team_1_prob_vegas: number | null;
  team_2_prob_vegas: number | null;
  team_1_ml: number | null;
  team_2_ml: number | null;
  total_pred: number | null;
  vegas_ou_line: number | null;
  actual_total: number | null;
  actual_winner: string | null;
}

export interface GamesResponse {
  date: string;
  games: Game[];
  count: number;
}

export async function fetchGamesByDate(date: string): Promise<GamesResponse> {
  const res = await fetch(`/api/ncaamb/games?date=${date}`);
  if (!res.ok) throw new Error('Failed to fetch games');
  return res.json();
}

export interface BookPerformance {
  date: string;
  book: string;
  ml_right: number;
  ml_total: number;
  ou_mae: number | null;
  ou_games: number;
  ap_ou_right: number | null;
  ap_ou_total: number | null;
  ap_ou_acc: number | null;
  ap_over_acc: number | null;
  ap_under_acc: number | null;
}

export interface PerformanceResponse {
  date: string;
  books: BookPerformance[];
}

export async function fetchPerformance(): Promise<PerformanceResponse> {
  const res = await fetch('/api/ncaamb/performance');
  if (!res.ok) throw new Error('Failed to fetch performance');
  return res.json();
}

export interface BracketGame {
  id: number;
  bracket_slot: string;
  round: string;
  region: string | null;
  pred_team_1: string;
  pred_team_1_seed: number | null;
  pred_team_2: string;
  pred_team_2_seed: number | null;
  prob_lgb: number | null;
  prob_xgb: number | null;
  prob_logistic: number | null;
  prob_ensemble: number | null;
  predicted_winner: string;
  predicted_winner_seed: number | null;
  game_id: string | null;
  actual_team_1: string | null;
  actual_team_2: string | null;
  actual_winner: string | null;
  correct: number | null;
}

export interface BracketResponse {
  year: number;
  games: BracketGame[];
  count: number;
}

export async function fetchBracket(year = 2026): Promise<BracketResponse> {
  const res = await fetch(`/api/ncaamb/bracket?year=${year}`);
  if (!res.ok) throw new Error('Failed to fetch bracket');
  return res.json();
}
