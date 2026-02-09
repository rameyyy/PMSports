import type { Game } from '../../../api/ncaamb';

const formatOdds = (odds: number | null) => {
  if (odds === null) return '-';
  return odds > 0 ? `+${odds}` : `${odds}`;
};

interface GamesTableProps {
  games: Game[];
  loading: boolean;
  isToday: boolean;
}

export default function GamesTable({ games, loading, isToday }: GamesTableProps) {
  if (loading) {
    return (
      <div className="text-center text-slate-400 py-12">
        <p>Loading games...</p>
      </div>
    );
  }

  if (games.length === 0) {
    const isPre8am = isToday && new Date().getHours() < 8;
    return (
      <div className="text-center text-slate-400 py-12">
        <p>{isPre8am ? "Today's pipeline hasn't run yet. Check back after 8am." : 'No games found for this date.'}</p>
      </div>
    );
  }

  const isPast = !isToday;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm font-medium border-collapse border border-slate-600">
        <thead>
          <tr className="bg-slate-700 text-slate-200 font-semibold">
            <th className="px-3 py-1.5 border border-slate-600 text-left">MATCHUP</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">ALGOPICKS</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">TOTAL PRED</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">O/U LINE</th>
            {isPast && <th className="px-3 py-1.5 border border-slate-600 text-center">ACTUAL TOTAL</th>}
          </tr>
        </thead>

        <tbody>
          {games.map((game, index) => {
            const isNeutral = game.team_1_hna === 'N';
            const isTeam1Home = game.team_1_hna === 'H';
            const left = isTeam1Home ? game.team_2 : game.team_1;
            const right = isTeam1Home ? game.team_1 : game.team_2;
            const leftProb = isTeam1Home ? game.team_2_prob_algopicks : game.team_1_prob_algopicks;
            const rightProb = isTeam1Home ? game.team_1_prob_algopicks : game.team_2_prob_algopicks;
            const leftML = isTeam1Home ? game.team_2_ml : game.team_1_ml;
            const rightML = isTeam1Home ? game.team_1_ml : game.team_2_ml;
            const leftRank = isTeam1Home ? game.team_2_rank : game.team_1_rank;
            const rightRank = isTeam1Home ? game.team_1_rank : game.team_2_rank;
            const leftConf = isTeam1Home ? game.team_2_conference : game.team_1_conference;
            const rightConf = isTeam1Home ? game.team_1_conference : game.team_2_conference;

            const algoTied = leftProb !== null && rightProb !== null && leftProb === rightProb;
            const pickedName = algoTied ? null : ((rightProb ?? 0) >= (leftProb ?? 0) ? right : left);
            const pickedPct = algoTied ? null : ((rightProb ?? 0) >= (leftProb ?? 0) ? rightProb : leftProb);

            // For past games: green if correct, red if wrong (tied picks are always wrong)
            const pickCorrect = isPast && game.actual_winner !== null
              ? (pickedName !== null && game.actual_winner === pickedName)
              : null;

            // Total prediction: model pred vs vegas line
            const totalCorrect = isPast && game.actual_total !== null && game.total_pred !== null && game.vegas_ou_line !== null
              ? (game.total_pred > game.vegas_ou_line) === (game.actual_total > game.vegas_ou_line)
              : null;

            return (
              <tr
                key={game.game_id}
                className={`${index % 2 === 0 ? 'bg-slate-900' : 'bg-slate-800'} hover:bg-slate-700 transition-colors`}
              >
                {/* Matchup */}
                <td className="px-3 py-1.5 border border-slate-700 text-slate-200 max-w-[270px]">
                  {leftConf && rightConf && leftConf === rightConf && (
                    <span className="text-slate-500 text-xs">[{leftConf}] </span>
                  )}
                  {leftRank && <span className="text-amber-700 text-xs">#{leftRank} </span>}
                  <span className={isPast && game.actual_winner === left ? 'text-orange-400 font-semibold' : ''}>{left}</span>
                  {leftConf && leftConf !== rightConf && <span className="text-slate-500 text-xs"> [{leftConf}]</span>}
                  <span className="text-slate-500 text-xs"> ({formatOdds(leftML)})</span>
                  <span className="text-slate-500 mx-1">{isNeutral ? 'v' : '@'}</span>
                  {rightRank && <span className="text-amber-700 text-xs">#{rightRank} </span>}
                  <span className={isPast && game.actual_winner === right ? 'text-orange-400 font-semibold' : ''}>{right}</span>
                  {rightConf && leftConf !== rightConf && <span className="text-slate-500 text-xs"> [{rightConf}]</span>}
                  <span className="text-slate-500 text-xs"> ({formatOdds(rightML)})</span>
                </td>

                {/* AlgoPicks */}
                <td className="px-0.5 py-1.5 border border-slate-700 text-center whitespace-nowrap">
                  {pickedName !== null && pickedPct !== null ? (
                    <span className={`font-semibold ${
                      pickCorrect === null ? 'text-white' :
                      pickCorrect ? 'text-green-400' : 'text-red-400'
                    }`}>{pickedName} ({pickedPct.toFixed(1)}%)</span>
                  ) : (
                    <span className={`font-semibold ${isPast && game.actual_winner !== null ? 'text-red-400' : 'text-slate-500'}`}>
                      {isPast && game.actual_winner !== null ? 'None' : '-'}
                    </span>
                  )}
                </td>

                {/* Total Pred */}
                <td className="px-3 py-1.5 border border-slate-700 text-center whitespace-nowrap">
                  {game.total_pred !== null && game.vegas_ou_line !== null ? (
                    <>
                      <span className={`font-semibold ${isPast && game.actual_total !== null && game.total_pred === game.actual_total ? 'text-yellow-400' : 'text-orange-400'}`}>
                        {game.total_pred}{isPast && game.actual_total !== null && game.total_pred === game.actual_total && ' ★'}
                      </span>
                      {game.total_pred !== game.vegas_ou_line && (
                        <span className="text-slate-500 ml-1">({game.total_pred > game.vegas_ou_line ? 'O' : 'U'})</span>
                      )}
                    </>
                  ) : game.total_pred !== null ? (
                    <span className="text-slate-300">{game.total_pred}</span>
                  ) : (
                    <span className="text-slate-500">-</span>
                  )}
                </td>

                {/* Line */}
                <td className="px-3 py-1.5 border border-slate-700 text-center text-slate-300">
                  {game.vegas_ou_line ?? '-'}
                </td>

                {/* Actual Total - only for past days */}
                {isPast && (
                  <td className="px-3 py-1.5 border border-slate-700 text-center whitespace-nowrap">
                    {game.actual_total !== null ? (
                      game.total_pred === game.actual_total ? (
                        <span className="font-semibold text-yellow-400">{game.actual_total} &#9733;</span>
                      ) : (
                        <span className={`font-semibold ${totalCorrect ? 'text-green-400' : 'text-red-400'}`}>
                          {game.actual_total}
                        </span>
                      )
                    ) : (
                      <span className="text-slate-500">-</span>
                    )}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
