import { useState, useEffect } from 'react';
import type { Bet, BettingStats } from './types';
import {
  formatDateShort,
  formatDateLong,
  getBookmakerDisplayName,
  formatNetProfit
} from './utils';
import { fetchBetsAndStats } from './api';
import type { PaginationInfo } from './api';

interface EventGroup {
  event_name: string;
  fight_date: string;
  bets: Bet[];
  total_stake: number;
  potential_profit: number;
  net_profit: number;
  bet_count: number;
  bets_won: number;
  bets_lost: number;
  bets_void: number;
  has_settled_bets: boolean;
}

function groupBetsByEvent(bets: Bet[]): EventGroup[] {
  const grouped = bets.reduce((acc, bet) => {
    const key = `${bet.event_name}-${bet.fight_date}`;
    if (!acc[key]) {
      acc[key] = {
        event_name: bet.event_name,
        fight_date: bet.fight_date,
        bets: [],
        total_stake: 0,
        potential_profit: 0,
        net_profit: 0,
        bet_count: 0,
        bets_won: 0,
        bets_lost: 0,
        bets_void: 0,
        has_settled_bets: false
      };
    }
    acc[key].bets.push(bet);
    acc[key].total_stake += Number(bet.stake ?? 0);
    acc[key].potential_profit += Number(bet.potential_profit ?? 0);
    acc[key].bet_count += 1;
    
    // Calculate net profit and count outcomes for settled bets
    if (bet.bet_outcome === 'won') {
      acc[key].net_profit += Number(bet.potential_profit ?? 0);
      acc[key].bets_won += 1;
      acc[key].has_settled_bets = true;
    } else if (bet.bet_outcome === 'lost') {
      acc[key].net_profit -= Number(bet.stake ?? 0);
      acc[key].bets_lost += 1;
      acc[key].has_settled_bets = true;
    } else if (bet.bet_outcome === 'void') {
      acc[key].bets_void += 1;
      acc[key].has_settled_bets = true;
    }
    
    return acc;
  }, {} as { [key: string]: EventGroup });

  return Object.values(grouped).sort((a, b) => 
    new Date(b.fight_date).getTime() - new Date(a.fight_date).getTime()
  );
}

export default function Bets() {
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const [expandedBet, setExpandedBet] = useState<string | null>(null);
  const [bets, setBets] = useState<Bet[]>([]);
  const [stats, setStats] = useState<BettingStats | null>(null);
  const [pagination, setPagination] = useState<PaginationInfo | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [statsLoaded, setStatsLoaded] = useState(false);
  const [betsLoaded, setBetsLoaded] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pendingPage, setPendingPage] = useState(1);
  const [betsLoading, setBetsLoading] = useState(false);
  const eventsPerPage = 10;

  // Fetch betting stats once on mount
  useEffect(() => {
    const loadStats = async () => {
      try {
        const res = await fetch('/api/ufc/bets/stats');
        if (!res.ok) throw new Error('Failed to fetch betting stats');
        const statsData = await res.json();
        setStats(statsData);
      } catch (error) {
        console.error('Error fetching betting stats:', error);
      } finally {
        setStatsLoaded(true);
      }
    };

    loadStats();
  }, []);

  // Fetch paginated bets when pending page changes
  useEffect(() => {
    const loadBets = async () => {
      setBetsLoading(true);
      try {
        const { bets: betsData, pagination: paginationData } = await fetchBetsAndStats(pendingPage, eventsPerPage);
        setBets(betsData);
        setPagination(paginationData);
        // Only update the displayed page number AFTER data loads
        setCurrentPage(pendingPage);
      } catch (error) {
        console.error('Error fetching bets:', error);
      } finally {
        setBetsLoading(false);
        setBetsLoaded(true);
      }
    };

    loadBets();
  }, [pendingPage]);

  // Set initial loading to false once both stats and bets are loaded
  useEffect(() => {
    if (statsLoaded && betsLoaded) {
      setInitialLoading(false);
    }
  }, [statsLoaded, betsLoaded]);

  if (initialLoading) {
    return (
      <div className="w-full p-8 flex items-center justify-center">
        <p className="text-white text-lg">Loading bets...</p>
      </div>
    );
  }

  const eventGroups = groupBetsByEvent(bets);

  // Use backend pagination data (pagination is already by events)
  const totalPages = pagination?.total_pages || 1;

  const toggleEvent = (eventKey: string) => {
    setExpandedEvent(expandedEvent === eventKey ? null : eventKey);
  };

  const toggleBet = (betKey: string) => {
    setExpandedBet(expandedBet === betKey ? null : betKey);
  };

  const netProfit = Number(stats?.total_profit ?? 0) - Number(stats?.total_loss ?? 0);
  const totalStaked = Number(stats?.total_staked ?? 0);

  return (
    <div className="w-full p-4 sm:p-8">     
      {/* Stats Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 sm:gap-4 mb-8 sm:mb-12">
        <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
          <p className="text-slate-400 text-xs sm:text-sm mb-1">Total Bets</p>
          <p className="text-white text-lg sm:text-2xl font-bold">{(stats?.bets_won || 0) + (stats?.bets_lost || 0) }</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
          <p className="text-slate-400 text-xs sm:text-sm mb-1">Total Staked</p>
          <p className="text-white text-lg sm:text-2xl font-bold">${totalStaked.toFixed(2)}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
          <p className="text-slate-400 text-xs sm:text-sm mb-1">Total Won</p>
          <p className="text-green-400 text-lg sm:text-2xl font-bold">${Number(stats?.total_profit ?? 0).toFixed(2)}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
          <p className="text-slate-400 text-xs sm:text-sm mb-1">Net Profit</p>
          <p className={`text-lg sm:text-2xl font-bold ${netProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatNetProfit(netProfit)}
          </p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
          <p className="text-slate-400 text-xs sm:text-sm mb-1">Win Rate</p>
          <p className="text-white text-lg sm:text-2xl font-bold">{Number(stats?.win_rate ?? 0).toFixed(1)}%</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
          <p className="text-slate-400 text-xs sm:text-sm mb-1">ROI</p>
          <p className={`text-lg sm:text-2xl font-bold ${Number(stats?.roi ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {Number(stats?.roi ?? 0) >= 0 ? '+' : ''}{Number(stats?.roi ?? 0).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Events List */}
      <div className="space-y-6">
        {eventGroups.map((eventGroup) => {
          const eventKey = `${eventGroup.event_name}-${eventGroup.fight_date}`;
          const isEventExpanded = expandedEvent === eventKey;
          
          return (
            <div key={eventKey} className="bg-slate-800/30 rounded-lg border border-slate-700">
              {/* Event Header */}
              <button
                onClick={() => toggleEvent(eventKey)}
                className="w-full p-4 sm:p-6 hover:bg-slate-800/50 transition-colors rounded-lg"
              >
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div className="text-left">
                    <h2 className="text-base sm:text-xl font-bold text-white mb-1">{eventGroup.event_name}</h2>
                    <p className="text-xs sm:text-sm text-slate-400">{formatDateLong(eventGroup.fight_date)}</p>
                  </div>
                  <div className="flex flex-wrap items-center gap-3 sm:gap-8">
                    {/* Show breakdown if event has settled bets */}
                    {eventGroup.has_settled_bets ? (
                      <>
                        <div className="text-right">
                          <p className="text-slate-400 text-xs mb-1">Won</p>
                          <p className="text-green-400 text-sm sm:text-lg font-semibold">{eventGroup.bets_won}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-slate-400 text-xs mb-1">Lost</p>
                          <p className="text-red-400 text-sm sm:text-lg font-semibold">{eventGroup.bets_lost}</p>
                        </div>
                        {eventGroup.bets_void > 0 && (
                          <div className="text-right">
                            <p className="text-slate-400 text-xs mb-1">Void</p>
                            <p className="text-slate-400 text-sm sm:text-lg font-semibold">{eventGroup.bets_void}</p>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="text-right">
                        <p className="text-slate-400 text-xs mb-1">Bets</p>
                        <p className="text-white text-sm sm:text-lg font-semibold">{eventGroup.bet_count}</p>
                      </div>
                    )}
                    <div className="text-right hidden sm:block">
                      <p className="text-slate-400 text-xs mb-1">Total Staked</p>
                      <p className="text-white text-sm sm:text-lg font-semibold">${eventGroup.total_stake.toFixed(2)}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-slate-400 text-xs mb-1">
                        {eventGroup.has_settled_bets ? 'Profit' : 'Potential'}
                      </p>
                      <p className={`text-sm sm:text-lg font-semibold ${
                        eventGroup.has_settled_bets
                          ? (eventGroup.net_profit >= 0 ? 'text-green-400' : 'text-red-400')
                          : 'text-green-400'
                      }`}>
                        {eventGroup.has_settled_bets
                          ? formatNetProfit(eventGroup.net_profit)
                          : `$${eventGroup.potential_profit.toFixed(2)}`
                        }
                      </p>
                    </div>
                    <svg
                      className={`w-5 h-5 sm:w-6 sm:h-6 text-slate-400 transition-transform flex-shrink-0`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isEventExpanded ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                    </svg>
                  </div>
                </div>
              </button>

              {/* Event Bets */}
              {isEventExpanded && (
                <div className="px-4 sm:px-6 pb-4 sm:pb-6 space-y-3 sm:space-y-4">
                  {eventGroup.bets.map((bet, index) => {
                    const betKey = `${bet.fight_date}-${bet.fighter1_name}-${index}`;
                    const isBetExpanded = expandedBet === betKey;
                    const statusColor = bet.bet_outcome === 'won' ? 'text-green-400' : 
                                       bet.bet_outcome === 'lost' ? 'text-red-400' : 
                                       'text-orange-400';
                    
                    const fighterBetOn = bet.fighter_bet_on === '0' ? bet.fighter1_name : bet.fighter2_name;
                    const oddsValue = bet.fighter_bet_on === '0' ? bet.fighter1_odds : bet.fighter2_odds;
                    const evValue = Number(bet.fighter_bet_on === '0' ? (bet.fighter1_ev ?? 0) : (bet.fighter2_ev ?? 0));
                    const predValue = Number(bet.fighter_bet_on === '0' ? (bet.fighter1_pred ?? 0) : (bet.fighter2_pred ?? 0));
                    
                    return (
                      <div key={betKey} className="w-full">
                        {/* Bet header */}
                        <div className="bg-slate-700/30 rounded-lg p-3 sm:p-4 hover:bg-slate-700/40 transition-colors">
                          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-2">
                            <div>
                              <h3 className="text-sm sm:text-base font-semibold text-white">
                                {bet.fighter1_name} vs {bet.fighter2_name}
                              </h3>
                            </div>
                            <div className="text-right">
                              <p className={`text-base sm:text-lg font-bold ${statusColor}`}>
                                {bet.bet_outcome.toUpperCase()}
                              </p>
                            </div>
                          </div>

                          {/* Quick stats */}
                          <div className="grid grid-cols-2 sm:flex sm:items-center sm:justify-between gap-2 text-xs sm:text-sm">
                            <span className="text-slate-300">Bet On: <span className="font-semibold text-white">{fighterBetOn}</span></span>
                            <span className="text-slate-300">
                              Odds: <span className="font-semibold text-white">
                                {Number(oddsValue) >= 0 ? `+${oddsValue}` : oddsValue}
                              </span>
                            </span>
                            <span className="text-slate-300">Stake: <span className="font-semibold text-white">${Number(bet.stake ?? 0).toFixed(2)}</span></span>
                            <span className="text-slate-300">To Win: <span className="font-semibold text-green-400">${Number(bet.potential_profit ?? 0).toFixed(2)}</span></span>
                          </div>
                        </div>

                        {/* Expand button */}
                        <div className="flex items-center justify-center mt-2">
                          <button
                            onClick={() => toggleBet(betKey)}
                            className="text-slate-400 text-xs hover:text-slate-300 transition-colors flex items-center gap-1"
                          >
                            <span>{isBetExpanded ? 'Hide' : 'See'} Details</span>
                            <svg
                              className={`w-4 h-4 transition-transform ${isBetExpanded ? 'rotate-180' : ''}`}
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                          </button>
                        </div>

                        {/* Expanded details */}
                        {isBetExpanded && (
                          <div className="mt-3 bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 sm:gap-4">
                              <div>
                                <p className="text-slate-400 text-xs mb-1">Sportsbook</p>
                                <p className="text-white text-sm sm:text-lg font-semibold">{getBookmakerDisplayName(bet.sportsbook)}</p>
                              </div>
                              <div>
                                <p className="text-slate-400 text-xs mb-1">Bet Type</p>
                                <p className="text-white text-sm sm:text-lg font-semibold">
                                  {bet.bet_type === 'moneyline' ? 'MoneyLine' : bet.bet_type}
                                </p>
                              </div>
                              <div>
                                <p className="text-slate-400 text-xs mb-1">Expected Value</p>
                                <p className={`text-sm sm:text-lg font-semibold ${evValue > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  {evValue > 0 ? '+' : ''}{evValue.toFixed(2)}%
                                </p>
                              </div>
                              <div>
                                <p className="text-slate-400 text-xs mb-1">Model Confidence</p>
                                <p className="text-white text-sm sm:text-lg font-semibold">{predValue.toFixed(2)}%</p>
                              </div>
                              <div>
                                <p className="text-slate-400 text-xs mb-1">Bet Date</p>
                                <p className="text-white text-sm sm:text-lg font-semibold">{formatDateShort(bet.bet_date)}</p>
                              </div>
                              <div>
                                <p className="text-slate-400 text-xs mb-1">Amount Staked</p>
                                <p className="text-white text-sm sm:text-lg font-semibold">${Number(bet.stake ?? 0).toFixed(2)}</p>
                              </div>
                              <div>
                                <p className="text-slate-400 text-xs mb-1">Potential Profit</p>
                                <p className="text-green-400 text-sm sm:text-lg font-semibold">${Number(bet.potential_profit ?? 0).toFixed(2)}</p>
                              </div>
                              <div>
                                <p className="text-slate-400 text-xs mb-1">Actual Result</p>
                                <p className={`text-sm sm:text-lg font-semibold ${bet.bet_outcome === 'won' ? 'text-green-400' : bet.bet_outcome === 'lost' ? 'text-red-400' : 'text-slate-400'}`}>
                                  {bet.bet_outcome === 'won' ? `+$${Number(bet.potential_profit ?? 0).toFixed(2)}` :
                                   bet.bet_outcome === 'lost' ? `-$${Number(bet.potential_loss ?? 0).toFixed(2)}` :
                                   'Pending'}
                                </p>
                              </div>
                            </div>

                            {/* Both fighters' odds for reference */}
                            <div className="mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-slate-700">
                              <p className="text-slate-400 text-xs mb-2">Fight Odds</p>
                              <div className="grid grid-cols-2 gap-3 sm:gap-4">
                                <div>
                                  <p className="text-slate-300 text-xs sm:text-sm">{bet.fighter1_name}</p>
                                  <p className="text-white text-xs sm:text-sm font-semibold">{bet.fighter1_odds} (EV: {Number(bet.fighter1_ev ?? 0).toFixed(2)}%)</p>
                                </div>
                                <div>
                                  <p className="text-slate-300 text-xs sm:text-sm">{bet.fighter2_name}</p>
                                  <p className="text-white text-xs sm:text-sm font-semibold">{bet.fighter2_odds} (EV: {Number(bet.fighter2_ev ?? 0).toFixed(2)}%)</p>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 sm:gap-4 mt-8">
          <button
            onClick={() => setPendingPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1 || betsLoading}
            className="px-3 sm:px-4 py-2 text-sm sm:text-base bg-slate-700 text-white rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Previous
          </button>
          <span className="text-xs sm:text-base text-slate-300 flex items-center gap-2">
            Page {currentPage} of {totalPages}
            {betsLoading && (
              <svg className="w-4 h-4 animate-spin text-orange-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            )}
          </span>
          <button
            onClick={() => setPendingPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages || betsLoading}
            className="px-3 sm:px-4 py-2 text-sm sm:text-base bg-slate-700 text-white rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}