import { useState, useEffect } from 'react';

interface Bet {
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

interface BettingStats {
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

function calculateROI(total_staked: number, total_profit: number): number | null {
  if (total_staked === null || total_staked === 0 || total_profit === null) {
    return null;
  }
  return Number((total_staked + total_profit) / total_staked) * 100 - 100;
}

function getBookmakerDisplayName(key: string): string {
  const names: { [key: string]: string } = {
    'bovada': 'Bovada',
    'fanduel': 'FanDuel',
    'draftkings': 'DraftKings',
    'betmgm': 'BetMGM',
    'betonlineag': 'BetOnline',
    'betus': 'BetUS',
    'betrivers': 'BetRivers'
  };
  return names[key] || key;
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const year = date.getFullYear();
  return `${month}-${day}-${year}`;
}

export default function Bets() {
  const [expandedBet, setExpandedBet] = useState<string | null>(null);
  const [bets, setBets] = useState<Bet[]>([]);
  const [stats, setStats] = useState<BettingStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBetsAndStats = async () => {
      try {
        const [betsRes, statsRes] = await Promise.all([
          fetch('/api/ufc/bets'),
          fetch('/api/ufc/bets/stats')
        ]);
        
        const betsData = await betsRes.json();
        const statsData = await statsRes.json();
        
        setBets(betsData);
        setStats(statsData);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching bets:', error);
        setLoading(false);
      }
    };

    fetchBetsAndStats();
  }, []);

  if (loading) {
    return (
      <div className="w-full p-8 flex items-center justify-center">
        <p className="text-white text-lg">Loading bets...</p>
      </div>
    );
  }

  const toggleBet = (betKey: string) => {
    setExpandedBet(expandedBet === betKey ? null : betKey);
  };

  const netProfit = (stats?.total_profit ?? 0) - (stats?.total_loss ?? 0);
  const totalWon = (stats?.total_profit ?? 0) + Math.abs(stats?.total_loss ?? 0);

  return (
    <div className="w-full p-8">     
      {/* Stats Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-12">
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <p className="text-slate-400 text-sm mb-1">Total Bets</p>
          <p className="text-white text-2xl font-bold">{stats?.total_bets || 0}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <p className="text-slate-400 text-sm mb-1">Total Staked</p>
          <p className="text-white text-2xl font-bold">${Number(stats?.total_staked ?? 0).toFixed(2)}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <p className="text-slate-400 text-sm mb-1">Total Won</p>
          <p className="text-green-400 text-2xl font-bold">${Number(totalWon).toFixed(2)}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <p className="text-slate-400 text-sm mb-1">Net Profit</p>
          <p className={`text-2xl font-bold ${netProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${netProfit >= 0 ? '' : ''}{Number(netProfit).toFixed(2)}
          </p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <p className="text-slate-400 text-sm mb-1">Win Rate</p>
          <p className="text-white text-2xl font-bold">{Number(stats?.win_rate ?? 0).toFixed(1)}%</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
          <p className="text-slate-400 text-sm mb-1">ROI</p>
          <p className={`text-2xl font-bold ${(calculateROI(Number(stats?.total_staked), Number(netProfit)) ?? 0) >= 1 ? 'text-green-400' : 'text-red-400'}`}>
            {((calculateROI(Number(stats?.total_staked), Number(netProfit)) ?? 1)).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Bets List */}
      <div className="space-y-4">
        {bets.map((bet, index) => {
          const betKey = `${bet.fight_date}-${bet.fighter1_name}-${index}`;
          const isExpanded = expandedBet === betKey;
          const statusColor = bet.bet_outcome === 'won' ? 'text-green-400' : 
                             bet.bet_outcome === 'lost' ? 'text-red-400' : 
                             'text-orange-400';
          
          // Determine which fighter was bet on and get their odds
          const fighterBetOn = bet.fighter_bet_on === '0' ? bet.fighter1_name : bet.fighter2_name;
          const oddsValue = bet.fighter_bet_on === '0' ? bet.fighter1_odds : bet.fighter2_odds;
          const evValue = Number(bet.fighter_bet_on === '0' ? (bet.fighter1_ev ?? 0) : (bet.fighter2_ev ?? 0));
          const predValue = Number(bet.fighter_bet_on === '0' ? (bet.fighter1_pred ?? 0) : (bet.fighter2_pred ?? 0));
          
          return (
            <div key={betKey} className="w-full">
              {/* Bet header */}
              <div className="bg-slate-700/30 rounded-lg p-4 hover:bg-slate-700/40 transition-colors">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h3 className="text-base font-semibold text-white">
                      {bet.fighter1_name} vs {bet.fighter2_name}
                    </h3>
                    <p className="text-sm text-slate-400">{bet.event_name}</p>
                  </div>
                  <div className="text-right">
                    <p className={`text-lg font-bold ${statusColor}`}>
                      {bet.bet_outcome.toUpperCase()}
                    </p>
                    <p className="text-sm text-slate-400">{formatDate(bet.fight_date)}</p>
                  </div>
                </div>

                {/* Quick stats */}
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-300">Bet On: <span className="font-semibold text-white">{fighterBetOn}</span></span>
                  <span className="text-slate-300">Odds: <span className="font-semibold text-white">+{oddsValue}</span></span>
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
                  <span>{isExpanded ? 'Hide' : 'See'} Details</span>
                  <svg
                    className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>

              {/* Expanded details */}
              {isExpanded && (
                <div className="mt-3 bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Sportsbook</p>
                      <p className="text-white text-lg font-semibold">{getBookmakerDisplayName(bet.sportsbook)}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Bet Type</p>
                      <p className="text-white text-lg font-semibold">
                        {bet.bet_type === 'moneyline' ? 'MoneyLine' : bet.bet_type}
                        </p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Expected Value</p>
                      <p className={`text-lg font-semibold ${evValue > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {evValue > 0 ? '+' : ''}{evValue.toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Model Confidence</p>
                      <p className="text-white text-lg font-semibold">{predValue.toFixed(2)}%</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Bet Date</p>
                      <p className="text-white text-lg font-semibold">{formatDate(bet.bet_date)}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Amount Staked</p>
                      <p className="text-white text-lg font-semibold">${Number(bet.stake ?? 0).toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Potential Profit</p>
                      <p className="text-green-400 text-lg font-semibold">${Number(bet.potential_profit ?? 0).toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Actual Result</p>
                      <p className={`text-lg font-semibold ${bet.bet_outcome === 'won' ? 'text-green-400' : bet.bet_outcome === 'lost' ? 'text-red-400' : 'text-slate-400'}`}>
                        {bet.bet_outcome === 'won' ? `+$${Number(bet.potential_profit ?? 0).toFixed(2)}` : 
                         bet.bet_outcome === 'lost' ? `-$${Number(bet.potential_loss ?? 0).toFixed(2)}` : 
                         'Pending'}
                      </p>
                    </div>
                  </div>
                  
                  {/* Both fighters' odds for reference */}
                  <div className="mt-4 pt-4 border-t border-slate-700">
                    <p className="text-slate-400 text-xs mb-2">Fight Odds</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-slate-300 text-sm">{bet.fighter1_name}</p>
                        <p className="text-white font-semibold">{bet.fighter1_odds} (EV: {Number(bet.fighter1_ev ?? 0).toFixed(2)}%)</p>
                      </div>
                      <div>
                        <p className="text-slate-300 text-sm">{bet.fighter2_name}</p>
                        <p className="text-white font-semibold">{bet.fighter2_odds} (EV: {Number(bet.fighter2_ev ?? 0).toFixed(2)}%)</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}