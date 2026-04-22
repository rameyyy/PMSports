export interface UFCNextEvent {
  name: string | null;
  date: string | null;
  days_away: number | null;
  fight_count: number;
}

export interface UFCAccuracy {
  accuracy: number;
  correct: number;
  total: number;
}

export interface UFCPickOfWeek {
  record: string;
  win_rate: number;
  total: number;
  avg_odds: number | null;
  units: number | null;
}

export interface UFCPick {
  matchup: string | null;
  event: string | null;
  prediction: string | null;
  odds: number | null;
  correct?: boolean | null;
}

export interface UFCHomepageStats {
  next_event: UFCNextEvent;
  model_accuracy: UFCAccuracy;
  vegas_accuracy: UFCAccuracy;
  pick_of_week: UFCPickOfWeek;
  last_pick: UFCPick | null;
  next_pick: UFCPick | null;
}

export async function fetchUFCHomepageStats(): Promise<UFCHomepageStats> {
  const res = await fetch('/api/ufc/homepage-stats');
  if (!res.ok) throw new Error('Failed to fetch UFC homepage stats');
  return res.json();
}
