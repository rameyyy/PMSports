import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { fetchRiskMetrics, fetchBetAnalytics } from './api';
import type { RiskMetrics, BetAnalytics } from './types';

// Utility function to format strategy names
const formatStrategyName = (strategy: string): string => {
  if (strategy === 'Flat_50') return 'Flat $50';
  const match = strategy.match(/Kelly_(\d+)pct/);
  if (match) {
    return `Kelly ${match[1]}%`;
  }
  return strategy;
};

export default function BetAnalyticsPage() {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('Kelly_5pct');
  const [betAnalytics, setBetAnalytics] = useState<BetAnalytics[]>([]);
  const [loading, setLoading] = useState(true);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [selectedTooltip, setSelectedTooltip] = useState<string | null>(null);

  const tooltips: { [key: string]: string } = {
    strategy: 'Strategy name & Kelly %',
    bets: 'Total bets placed',
    winrate: '% of bets won',
    profit: 'Net profit/loss',
    roi: 'Return on investment %',
    maxdd: 'Maximum Drawdown - The largest peak-to-trough decline experienced. How much your bankroll dropped at worst.',
    volatility: 'Volatility - Measures how much your returns swing up and down. Higher = more unpredictable, Lower = more stable.',
    sharpe: 'Sharpe Ratio - Risk-adjusted returns. Measures how much profit you make per unit of risk. Higher = better risk-adjusted performance.',
    kelly: 'Average Kelly Fraction - The average percentage of your bankroll bet per play. Shows how aggressively the strategy sizes bets.'
  };

  // Available strategies based on your kelly.py
  const strategies = [
    'Flat_50',
    'Kelly_5pct',
    'Kelly_6pct',
    'Kelly_7pct',
    'Kelly_8pct',
    'Kelly_9pct',
    'Kelly_10pct'
  ];

  // Load risk metrics on mount
  useEffect(() => {
    const loadRiskMetrics = async () => {
      try {
        const data = await fetchRiskMetrics();
        // Parse string numbers to actual numbers
        const parsedData = data.map(metric => ({
          ...metric,
          win_rate: parseFloat(metric.win_rate as any),
          total_profit: parseFloat(metric.total_profit as any),
          roi: parseFloat(metric.roi as any),
          max_drawdown: parseFloat(metric.max_drawdown as any),
          current_drawdown: parseFloat(metric.current_drawdown as any),
          sharpe_ratio: parseFloat(metric.sharpe_ratio as any),
          volatility: parseFloat(metric.volatility as any),
          avg_kelly_fraction: parseFloat(metric.avg_kelly_fraction as any),
          kelly_utilization: parseFloat(metric.kelly_utilization as any)
        }));
        setRiskMetrics(parsedData);
      } catch (error) {
        console.error('Error fetching risk metrics:', error);
      } finally {
        setLoading(false);
      }
    };
    loadRiskMetrics();
  }, []);

  // Load bet analytics when strategy changes
  useEffect(() => {
    const loadBetAnalytics = async () => {
      setAnalyticsLoading(true);
      try {
        const data = await fetchBetAnalytics(selectedStrategy);
        // Parse string numbers to actual numbers
        const parsedData = data.map(bet => ({
          ...bet,
          bet_size: parseFloat(bet.bet_size as any),
          win_probability: parseFloat(bet.win_probability as any),
          decimal_odds: parseFloat(bet.decimal_odds as any),
          kelly_fraction: parseFloat(bet.kelly_fraction as any),
          expected_value: parseFloat(bet.expected_value as any),
          bankroll_before: parseFloat(bet.bankroll_before as any),
          bankroll_after: parseFloat(bet.bankroll_after as any),
          cumulative_profit: parseFloat(bet.cumulative_profit as any),
          actual_profit: parseFloat(bet.actual_profit as any),
          running_roi: parseFloat(bet.running_roi as any),
          max_drawdown: parseFloat(bet.max_drawdown as any)
        }));
        setBetAnalytics(parsedData);
      } catch (error) {
        console.error('Error fetching bet analytics:', error);
        setBetAnalytics([]);
      } finally {
        setAnalyticsLoading(false);
      }
    };
    loadBetAnalytics();
  }, [selectedStrategy]);

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  // Format percentage
  const formatPercent = (value: number, isAlreadyPercent: boolean = false) => {
    if (isAlreadyPercent) {
      return `${value.toFixed(2)}%`;
    }
    return `${(value * 100).toFixed(2)}%`;
  };

  // Prepare chart data - use linear index (1, 2, 3...) instead of bet_sequence
  const bankrollChartData = betAnalytics.map((bet, index) => ({
    bet: index + 1,
    bankroll: bet.bankroll_after,
    profit: bet.cumulative_profit,
    roi: bet.running_roi
  }));

  // Prepare ROI chart data
  const roiChartData = betAnalytics.map((bet, index) => ({
    bet: index + 1,
    roi: bet.running_roi
  }));

  // Prepare drawdown chart data
  const drawdownChartData = betAnalytics.map((bet, index) => ({
    bet: index + 1,
    drawdown: -bet.max_drawdown
  }));

  // Get current strategy risk metrics
  const currentMetrics = riskMetrics.find(m => m.strategy_name === selectedStrategy);

  if (loading) {
    return (
      <div className="text-center text-white py-12">
        <div className="animate-pulse">Loading analytics...</div>
      </div>
    );
  }

  return (
    <div className="space-y-4 md:space-y-8">
      {/* Page Header */}
      <div className="text-center">
        <h1 className="text-2xl md:text-3xl font-bold text-white mb-2">Betting Analytics Dashboard</h1>
        <p className="text-sm md:text-base text-slate-400">Kelly Criterion and Flat Betting Performance Analysis</p>
      </div>

      {/* Risk Metrics Table */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
        <div className="px-4 md:px-6 py-3 md:py-4 border-b border-slate-700">
          <h2 className="text-lg md:text-xl font-bold text-white">Strategy Performance Summary</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs md:text-sm">
            <thead className="bg-slate-900/50">
              <tr>
                <th className="px-2 md:px-4 py-2 md:py-3 text-left text-slate-300 font-semibold sticky left-0 bg-slate-900/50">Strategy</th>
                <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold whitespace-nowrap cursor-pointer" onClick={() => setSelectedTooltip(selectedTooltip === 'bets' ? null : 'bets')}>Bets</th>
                <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold whitespace-nowrap cursor-pointer" onClick={() => setSelectedTooltip(selectedTooltip === 'winrate' ? null : 'winrate')}>Win Rate</th>
                <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold whitespace-nowrap cursor-pointer" onClick={() => setSelectedTooltip(selectedTooltip === 'profit' ? null : 'profit')}>Profit</th>
                <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold cursor-pointer" onClick={() => setSelectedTooltip(selectedTooltip === 'roi' ? null : 'roi')}>ROI</th>
                <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold cursor-pointer" onClick={() => setSelectedTooltip(selectedTooltip === 'maxdd' ? null : 'maxdd')}>Max DD</th>
                <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold cursor-pointer" onClick={() => setSelectedTooltip(selectedTooltip === 'volatility' ? null : 'volatility')}>Volatility</th>
                <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold cursor-pointer" onClick={() => setSelectedTooltip(selectedTooltip === 'sharpe' ? null : 'sharpe')}>Sharpe</th>
                <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold cursor-pointer" onClick={() => setSelectedTooltip(selectedTooltip === 'kelly' ? null : 'kelly')}>Avg Kelly</th>
              </tr>
            </thead>
            <tbody>
              {riskMetrics.filter(metric => metric.strategy_name !== 'Kelly_4pct' && metric.strategy_name !== 'Kelly_11pct').map((metric) => (
                <tr
                  key={metric.strategy_name}
                  className={`border-t border-slate-700 hover:bg-slate-700/30 cursor-pointer transition-colors ${
                    selectedStrategy === metric.strategy_name ? 'bg-orange-500/10' : ''
                  }`}
                  onClick={() => setSelectedStrategy(metric.strategy_name)}
                >
                  <td className="px-2 md:px-4 py-2 md:py-3 text-white font-semibold sticky left-0 bg-slate-800/50 whitespace-nowrap">
                    {formatStrategyName(metric.strategy_name)}
                  </td>
                  <td className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300">{metric.total_bets}</td>
                  <td className="px-2 md:px-4 py-2 md:py-3 text-right">
                    <span className={metric.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}>
                      {formatPercent(metric.win_rate, false)}
                    </span>
                  </td>
                  <td className="px-2 md:px-4 py-2 md:py-3 text-right">
                    <span className={metric.total_profit >= 0 ? 'text-green-400' : 'text-red-400'}>
                      ${metric.total_profit.toFixed(0)}
                    </span>
                  </td>
                  <td className="px-2 md:px-4 py-2 md:py-3 text-right">
                    <span className={metric.roi >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {metric.roi.toFixed(2)}%
                    </span>
                  </td>
                  <td className="px-2 md:px-4 py-2 md:py-3 text-right text-red-400">
                    ${metric.max_drawdown.toFixed(0)}
                  </td>
                  <td className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300">
                    {metric.volatility.toFixed(4)}
                  </td>
                  <td className="px-2 md:px-4 py-2 md:py-3 text-right">
                    <span className={metric.sharpe_ratio >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {metric.sharpe_ratio.toFixed(2)}
                    </span>
                  </td>
                  <td className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300">
                    {formatPercent(metric.avg_kelly_fraction, false)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {selectedTooltip && (
          <div className="mt-4 p-4 bg-orange-500/10 border border-orange-500/30 rounded-lg flex items-start gap-3">
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full border border-orange-400 text-orange-400 text-xs font-bold flex-shrink-0 mt-0.5">i</span>
            <p className="text-orange-400 font-semibold text-sm">{tooltips[selectedTooltip]}</p>
          </div>
        )}
      </div>

      {/* Strategy Selector */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4 md:p-6">
        <h2 className="text-lg md:text-xl font-bold text-white mb-3 md:mb-4">Select Strategy to Analyze</h2>
        <div className="flex flex-wrap gap-2">
          {strategies.map((strategy) => (
            <button
              key={strategy}
              onClick={() => setSelectedStrategy(strategy)}
              className={`px-3 md:px-4 py-1.5 md:py-2 text-sm md:text-base rounded-lg font-semibold transition-all ${
                selectedStrategy === strategy
                  ? 'bg-orange-500 text-white shadow-lg shadow-orange-500/50'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              {formatStrategyName(strategy)}
            </button>
          ))}
        </div>
      </div>

      {analyticsLoading ? (
        <div className="text-center text-white py-12">
          <div className="animate-pulse">Loading {formatStrategyName(selectedStrategy)} data...</div>
        </div>
      ) : betAnalytics.length === 0 ? (
        <div className="text-center text-slate-400 py-12">
          No data available for {formatStrategyName(selectedStrategy)}
        </div>
      ) : (
        <>
          {/* Current Strategy Stats */}
          {currentMetrics && betAnalytics.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
              <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3 md:p-4">
                <div className="text-slate-400 text-xs md:text-sm mb-1">Final Bankroll</div>
                <div className="text-lg md:text-2xl font-bold text-white">
                  ${betAnalytics[betAnalytics.length - 1]?.bankroll_after.toFixed(0) || '1000'}
                </div>
              </div>
              <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3 md:p-4">
                <div className="text-slate-400 text-xs md:text-sm mb-1">Total Profit</div>
                <div className={`text-lg md:text-2xl font-bold ${currentMetrics.total_profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${currentMetrics.total_profit.toFixed(0)}
                </div>
              </div>
              <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3 md:p-4">
                <div className="text-slate-400 text-xs md:text-sm mb-1">ROI</div>
                <div className={`text-lg md:text-2xl font-bold ${currentMetrics.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {currentMetrics.roi.toFixed(2)}%
                </div>
              </div>
              <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3 md:p-4">
                <div className="text-slate-400 text-xs md:text-sm mb-1">Win Rate</div>
                <div className={`text-lg md:text-2xl font-bold ${currentMetrics.win_rate >= 0.5 ? 'text-green-400' : 'text-orange-400'}`}>
                  {formatPercent(currentMetrics.win_rate)}
                </div>
              </div>
            </div>
          )}

          {/* Bankroll Growth Chart */}
          {bankrollChartData.length > 0 && (
          <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4 md:p-6">
            <h2 className="text-lg md:text-xl font-bold text-white mb-3 md:mb-4">Bankroll Growth - {formatStrategyName(selectedStrategy)}</h2>
            <ResponsiveContainer width="100%" height={300} className="md:!h-[400px]">
              <LineChart data={bankrollChartData} margin={{ bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="bet"
                  stroke="#94a3b8"
                  label={{ value: 'Bet Sequence', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                />
                <YAxis
                  stroke="#94a3b8"
                  label={{ value: 'Bankroll ($)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelStyle={{ color: '#e2e8f0' }}
                  itemStyle={{ color: '#f97316' }}
                />
                <Legend
                  wrapperStyle={{ color: '#94a3b8', paddingTop: '15px' }}
                  iconSize={14}
                />
                <Line
                  type="monotone"
                  dataKey="bankroll"
                  stroke="#f97316"
                  strokeWidth={3}
                  dot={false}
                  name="Bankroll"
                />
                <Line
                  type="monotone"
                  dataKey="profit"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                  name="Cumulative Profit"
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          )}

          {/* ROI Progression Chart */}
          {roiChartData.length > 0 && (
          <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4 md:p-6">
            <h2 className="text-lg md:text-xl font-bold text-white mb-3 md:mb-4">ROI Progression - {formatStrategyName(selectedStrategy)}</h2>
            <ResponsiveContainer width="100%" height={250} className="md:!h-[300px]">
              <AreaChart data={roiChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="bet"
                  stroke="#94a3b8"
                  label={{ value: 'Bet Sequence', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                />
                <YAxis
                  stroke="#94a3b8"
                  label={{ value: 'ROI (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelStyle={{ color: '#e2e8f0' }}
                  itemStyle={{ color: '#10b981' }}
                  formatter={(value: number) => `${value.toFixed(2)}%`}
                />
                <Area
                  type="monotone"
                  dataKey="roi"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.3}
                  name="ROI %"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          )}

          {/* Drawdown Chart */}
          {drawdownChartData.length > 0 && (
          <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4 md:p-6">
            <h2 className="text-lg md:text-xl font-bold text-white mb-3 md:mb-4">Drawdown History - {formatStrategyName(selectedStrategy)}</h2>
            <ResponsiveContainer width="100%" height={250} className="md:!h-[300px]">
              <AreaChart data={drawdownChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="bet"
                  stroke="#94a3b8"
                  label={{ value: 'Bet Sequence', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                />
                <YAxis
                  stroke="#94a3b8"
                  label={{ value: 'Drawdown ($)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  labelStyle={{ color: '#e2e8f0' }}
                  itemStyle={{ color: '#ef4444' }}
                  formatter={(value: number) => formatCurrency(Math.abs(value))}
                />
                <Area
                  type="monotone"
                  dataKey="drawdown"
                  stroke="#ef4444"
                  fill="#ef4444"
                  fillOpacity={0.3}
                  name="Drawdown"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          )}

          {/* Recent Bets Table */}
          {betAnalytics.length > 0 && (
          <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
            <div className="px-4 md:px-6 py-3 md:py-4 border-b border-slate-700">
              <h2 className="text-lg md:text-xl font-bold text-white">Recent Bets - {formatStrategyName(selectedStrategy)}</h2>
            </div>
            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-xs md:text-sm">
                <thead className="bg-slate-900/50 sticky top-0">
                  <tr>
                    <th className="px-2 md:px-4 py-2 md:py-3 text-left text-slate-300 font-semibold sticky left-0 bg-slate-900/50">Bet #</th>
                    <th className="px-2 md:px-4 py-2 md:py-3 text-left text-slate-300 font-semibold whitespace-nowrap">Date</th>
                    <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold whitespace-nowrap">Bet Size</th>
                    <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold hidden sm:table-cell">Odds</th>
                    <th className="px-2 md:px-4 py-2 md:py-3 text-center text-slate-300 font-semibold">Result</th>
                    <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold whitespace-nowrap">P/L</th>
                    <th className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 font-semibold hidden md:table-cell">Bankroll</th>
                  </tr>
                </thead>
                <tbody>
                  {betAnalytics.slice().reverse().map((bet, reversedIndex) => {
                    const linearBetNum = betAnalytics.length - reversedIndex;
                    return (
                      <tr key={bet.id} className="border-t border-slate-700 hover:bg-slate-700/30">
                        <td className="px-2 md:px-4 py-2 md:py-3 text-white font-semibold sticky left-0 bg-slate-800/50">{linearBetNum}</td>
                        <td className="px-2 md:px-4 py-2 md:py-3 text-slate-300 whitespace-nowrap">
                          {new Date(bet.bet_date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                        </td>
                        <td className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300">
                          ${bet.bet_size.toFixed(0)}
                        </td>
                        <td className="px-2 md:px-4 py-2 md:py-3 text-right text-slate-300 hidden sm:table-cell">
                          {bet.decimal_odds.toFixed(2)}
                        </td>
                        <td className="px-2 md:px-4 py-2 md:py-3 text-center">
                          <span
                            className={`px-1.5 md:px-2 py-0.5 md:py-1 rounded text-[10px] md:text-xs font-semibold ${
                              bet.bet_outcome === 'won'
                                ? 'bg-green-500/20 text-green-400'
                                : bet.bet_outcome === 'lost'
                                ? 'bg-red-500/20 text-red-400'
                                : 'bg-slate-500/20 text-slate-400'
                            }`}
                          >
                            {bet.bet_outcome === 'won' ? 'W' : bet.bet_outcome === 'lost' ? 'L' : bet.bet_outcome.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-2 md:px-4 py-2 md:py-3 text-right">
                          <span className={bet.actual_profit >= 0 ? 'text-green-400' : 'text-red-400'}>
                            ${bet.actual_profit.toFixed(0)}
                          </span>
                        </td>
                        <td className="px-2 md:px-4 py-2 md:py-3 text-right text-white font-semibold hidden md:table-cell">
                          ${bet.bankroll_after.toFixed(0)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          )}
        </>
      )}
    </div>
  );
}
