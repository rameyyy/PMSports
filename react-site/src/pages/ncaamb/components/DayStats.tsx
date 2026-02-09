import type { Game } from '../../../api/ncaamb';

interface DayStatsProps {
  games: Game[];
}

export default function DayStats({ games }: DayStatsProps) {
  // Only look at completed games
  const completed = games.filter(g => g.actual_winner !== null);
  if (completed.length === 0) return null;

  // ML accuracy
  let mlWins = 0;
  let mlLosses = 0;
  let totalConfidence = 0;

  for (const g of completed) {
    const prob1 = g.team_1_prob_algopicks ?? 0;
    const prob2 = g.team_2_prob_algopicks ?? 0;
    const pick = prob1 >= prob2 ? g.team_1 : g.team_2;
    const conf = Math.max(prob1, prob2);
    totalConfidence += conf;

    if (pick === g.actual_winner) mlWins++;
    else mlLosses++;
  }

  const mlTotal = mlWins + mlLosses;
  const mlAcc = mlTotal > 0 ? (mlWins / mlTotal) * 100 : 0;
  const avgConf = mlTotal > 0 ? totalConfidence / mlTotal : 0;

  // MAE calculations
  const gamesWithTotals = completed.filter(g => g.total_pred !== null && g.actual_total !== null);
  const gamesWithLine = completed.filter(g => g.vegas_ou_line !== null && g.actual_total !== null);

  const modelMAE = gamesWithTotals.length > 0
    ? gamesWithTotals.reduce((sum, g) => sum + Math.abs(g.total_pred! - g.actual_total!), 0) / gamesWithTotals.length
    : 0;

  const vegasMAE = gamesWithLine.length > 0
    ? gamesWithLine.reduce((sum, g) => sum + Math.abs(g.vegas_ou_line! - g.actual_total!), 0) / gamesWithLine.length
    : 0;

  // O/U accuracy
  const ouGames = completed.filter(g => g.total_pred !== null && g.vegas_ou_line !== null && g.actual_total !== null && g.total_pred !== g.vegas_ou_line);
  let ouCorrect = 0;
  let overCorrect = 0;
  let overTotal = 0;
  let underCorrect = 0;
  let underTotal = 0;

  for (const g of ouGames) {
    const predOver = g.total_pred! > g.vegas_ou_line!;
    const actualOver = g.actual_total! > g.vegas_ou_line!;

    if (predOver) {
      overTotal++;
      if (actualOver) { overCorrect++; ouCorrect++; }
    } else {
      underTotal++;
      if (!actualOver) { underCorrect++; ouCorrect++; }
    }
  }

  const ouAcc = ouGames.length > 0 ? (ouCorrect / ouGames.length) * 100 : 0;
  const overAcc = overTotal > 0 ? (overCorrect / overTotal) * 100 : 0;
  const underAcc = underTotal > 0 ? (underCorrect / underTotal) * 100 : 0;

  return (
    <div className="grid grid-cols-4 sm:grid-cols-8 gap-2 mb-3">
      {/* Model ML */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Model ML</div>
        <div className="text-sm sm:text-lg font-bold text-white">{mlAcc.toFixed(1)}%</div>
        <div className="text-[10px] sm:text-xs text-slate-500">{mlWins}-{mlLosses}</div>
      </div>

      {/* Avg Model Conf */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Avg Conf</div>
        <div className="text-sm sm:text-lg font-bold text-white">{avgConf.toFixed(1)}%</div>
      </div>

      {/* Model Totals MAE */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Model MAE</div>
        <div className="text-sm sm:text-lg font-bold text-white">{modelMAE.toFixed(1)}</div>
      </div>

      {/* Vegas Totals MAE */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Vegas MAE</div>
        <div className="text-sm sm:text-lg font-bold text-white">{vegasMAE.toFixed(1)}</div>
      </div>

      {/* O/U Accuracy */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">O/U</div>
        <div className="text-sm sm:text-lg font-bold text-white">{ouAcc.toFixed(1)}%</div>
        <div className="text-[10px] sm:text-xs text-slate-500">{ouCorrect}-{ouGames.length - ouCorrect}</div>
      </div>

      {/* Over Accuracy */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Over</div>
        <div className="text-sm sm:text-lg font-bold text-white">{overAcc.toFixed(1)}%</div>
        <div className="text-[10px] sm:text-xs text-slate-500">{overCorrect}-{overTotal - overCorrect}</div>
      </div>

      {/* Under Accuracy */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Under</div>
        <div className="text-sm sm:text-lg font-bold text-white">{underAcc.toFixed(1)}%</div>
        <div className="text-[10px] sm:text-xs text-slate-500">{underCorrect}-{underTotal - underCorrect}</div>
      </div>
    </div>
  );
}
