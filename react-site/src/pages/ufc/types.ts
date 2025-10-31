export type Fight = {
  fight_id: string;
  event_id: string;
  fighter1_id: string;
  fighter2_id: string;
  fighter1_name: string;
  fighter2_name: string;
  fighter1_nickname: string | null;
  fighter2_nickname: string | null;
  fighter1_img_link: string | null;
  fighter2_img_link: string | null;
  algopick_model: string | null;
  algopick_prediction: number | null;
  algopick_probability: number | null;
  window_sample: number | null;
  correct: number | null;
  date: string;
  end_time: string | null;
  weight_class: string;
  win_method: string | null;
  fight_type: string | null;
};

export type BookmakerOdds = {
  fight_id: string;
  bookmaker: string;
  fighter1_id: string;
  fighter2_id: string;
  fighter1_odds: number;
  fighter2_odds: number;
  fighter1_odds_percent: number;
  fighter2_odds_percent: number;
  fighter1_ev: number | null;
  fighter2_ev: number | null;
  vigor: number;
  algopick_prediction: number | null;
  algopick_probability: number | null;
};

export type Event = {
  event_id: string;
  event_url: string;
  title: string;
  event_datestr: string;
  location: string;
  date: string;
};

export interface Bet {
  bet_date: string;
  bet_outcome: 'pending' | 'won' | 'lost' | 'push' | 'void';
  bet_type: string;
  event_name: string;
  fight_date: string;
  fighter1_name: string;
  fighter2_name: string;
  fighter1_odds: string;
  fighter2_odds: string;
  fighter1_ev: number | null;
  fighter2_ev: number | null;
  fighter1_pred: number | null;
  fighter2_pred: number | null;
  fighter_bet_on: string;
  sportsbook: string;
  stake: number | null;
  potential_profit: number | null;
  potential_loss: number | null;
}

export interface BettingStats {
  total_bets: number;
  total_staked: number | null;
  total_profit: number | null;
  total_loss: number | null;
  bets_won: number;
  bets_lost: number;
  bets_pending: number;
  win_rate: number | null;
  roi: number | null;
}

export interface ModelAccuracy {
  model_name: string;
  total_predictions: number;
  correct_predictions: number;
  accuracy: number;
  avg_confidence: number;
  avg_sample_size: number;
}

export interface RiskMetrics {
  id: number;
  strategy_name: string;
  calculation_date: string;
  total_bets: number;
  win_rate: number;
  total_profit: number;
  roi: number;
  max_drawdown: number;
  current_drawdown: number;
  sharpe_ratio: number;
  volatility: number;
  avg_kelly_fraction: number;
  kelly_utilization: number;
  created_at: string;
}

export interface BetAnalytics {
  id: number;
  fight_id: string;
  bet_sequence: number;
  strategy_name: string;
  bet_size: number;
  win_probability: number;
  decimal_odds: number;
  kelly_fraction: number;
  expected_value: number;
  bankroll_before: number;
  bankroll_after: number;
  cumulative_profit: number;
  bet_outcome: 'won' | 'lost' | 'pending' | 'void';
  actual_profit: number;
  running_roi: number;
  max_drawdown: number;
  current_win_streak: number;
  current_loss_streak: number;
  bet_date: string;
  created_at: string;
}
