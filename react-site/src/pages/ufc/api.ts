import type { Fight, BookmakerOdds, Event, Bet, BettingStats, ModelAccuracy } from './types';

// ===== EVENT APIs =====
export async function fetchUpcomingEvents(): Promise<Event[]> {
  const res = await fetch('/api/ufc/events/upcoming');
  if (!res.ok) throw new Error('Failed to fetch upcoming events');
  return res.json();
}

export interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  total_pages: number;
}

export interface PaginatedEventResponse {
  events: Event[];
  pagination: PaginationInfo;
}

export async function fetchPastEvents(page: number = 1, limit: number = 10): Promise<PaginatedEventResponse> {
  const res = await fetch(`/api/ufc/events/past?page=${page}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch past events');
  return res.json();
}

// ===== FIGHT APIs =====
export async function fetchEventFights(eventId: string): Promise<Fight[]> {
  const res = await fetch(`/api/ufc/events/${eventId}/fights`);
  if (!res.ok) throw new Error(`Failed to fetch fights for event ${eventId}`);
  return res.json();
}

// ===== ODDS APIs =====
export async function fetchFightOdds(fightId: string): Promise<BookmakerOdds[]> {
  const res = await fetch(`/api/ufc/fights/${fightId}/odds`);
  if (!res.ok) return [];
  return res.json();
}

// Fetch odds for multiple fights in parallel
export async function fetchMultipleFightOdds(
  fights: Fight[]
): Promise<{ [key: string]: BookmakerOdds[] }> {
  const oddsPromises = fights.map((fight) =>
    fetchFightOdds(fight.fight_id)
      .then(odds => ({ fightId: fight.fight_id, odds }))
      .catch(() => ({ fightId: fight.fight_id, odds: [] }))
  );

  const allOdds = await Promise.all(oddsPromises);
  const oddsMap: { [key: string]: BookmakerOdds[] } = {};
  allOdds.forEach(({ fightId, odds }) => {
    oddsMap[fightId] = odds;
  });

  return oddsMap;
}

// ===== BET APIs =====
export interface PaginatedBetsResponse {
  bets: Bet[];
  pagination: PaginationInfo;
}

export async function fetchAllBets(page: number = 1, limit: number = 10): Promise<PaginatedBetsResponse> {
  const res = await fetch(`/api/ufc/bets?page=${page}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch bets');
  return res.json();
}

export async function fetchPendingBets(page: number = 1, limit: number = 10): Promise<PaginatedBetsResponse> {
  const res = await fetch(`/api/ufc/bets/pending?page=${page}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch pending bets');
  return res.json();
}

export async function fetchSettledBets(page: number = 1, limit: number = 10): Promise<PaginatedBetsResponse> {
  const res = await fetch(`/api/ufc/bets/settled?page=${page}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch settled bets');
  return res.json();
}

export async function fetchBettingStats(): Promise<BettingStats> {
  const res = await fetch('/api/ufc/bets/stats');
  if (!res.ok) throw new Error('Failed to fetch betting stats');
  return res.json();
}

export async function fetchBetsAndStats(page: number = 1, limit: number = 10): Promise<{ bets: Bet[]; pagination: PaginationInfo }> {
  const res = await fetch(`/api/ufc/bets?page=${page}&limit=${limit}`);

  if (!res.ok) throw new Error('Failed to fetch bets');

  const betsData = await res.json();

  return { bets: betsData.bets, pagination: betsData.pagination };
}

// ===== MODEL APIs =====
export async function fetchModelAccuracies(): Promise<ModelAccuracy[]> {
  const res = await fetch('/api/ufc/model-accuracies');
  if (!res.ok) throw new Error('Failed to fetch model accuracies');
  const data = await res.json();
  // Filter out AlgoPicks model
  return data.filter((model: ModelAccuracy) => model.model_name !== 'AlgoPicks');
}
