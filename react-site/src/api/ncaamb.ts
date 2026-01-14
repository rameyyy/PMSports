export interface HomepageStats {
  todays_games: number;
  model_accuracy: number;
  model_correct: number;
  model_total: number;
  vegas_accuracy: number;
  vegas_correct: number;
  vegas_total: number;
  edge: number;
}

export interface Pick {
  game_id: string;
  matchup: string;
  picked_team: string;
  picked_odds: number;
  betting_rule: string;
  date: string;
  time: string | null;
  result: 'W' | 'L' | null;
}

export interface PickOfDayData {
  today_pick: Pick | null;
  yesterday_pick: Pick | null;
  record: {
    correct: number;
    total: number;
    accuracy: number;
    roi: number;
    avg_odds: number;
  };
}

export async function fetchHomepageStats(): Promise<HomepageStats> {
  const res = await fetch('/api/ncaamb/homepage-stats');
  if (!res.ok) throw new Error('Failed to fetch homepage stats');
  return res.json();
}

export async function fetchPickOfDay(): Promise<PickOfDayData> {
  const res = await fetch('/api/ncaamb/pick-of-day');
  if (!res.ok) throw new Error('Failed to fetch pick of the day');
  return res.json();
}
