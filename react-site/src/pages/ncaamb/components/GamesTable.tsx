// Placeholder mock data - will be replaced with real API data later
// For past days, games have: actualTotal (final score total), winner ('home' | 'away')
const MOCK_GAMES_TODAY = [
  { id: '1', home: 'UNC', away: 'Duke', homeML: -150, awayML: +130, totalLine: 145.5, modelPick: 'home', modelTotal: 148.2, modelHomeWin: 62.5, modelAwayWin: 37.5, vegasHomeWin: 58.0, vegasAwayWin: 42.0 },
  { id: '2', home: 'Kentucky', away: 'Kansas', homeML: +120, awayML: -140, totalLine: 152.0, modelPick: 'away', modelTotal: 155.8, modelHomeWin: 44.2, modelAwayWin: 55.8, vegasHomeWin: 46.5, vegasAwayWin: 53.5 },
  { id: '3', home: 'UCLA', away: 'Gonzaga', homeML: -110, awayML: -110, totalLine: 148.5, modelPick: 'home', modelTotal: 146.1, modelHomeWin: 51.8, modelAwayWin: 48.2, vegasHomeWin: 50.0, vegasAwayWin: 50.0 },
  { id: '4', home: 'Michigan St', away: 'Purdue', homeML: null, awayML: null, totalLine: 139.0, modelPick: 'away', modelTotal: 141.5, modelHomeWin: 38.9, modelAwayWin: 61.1, vegasHomeWin: 40.2, vegasAwayWin: 59.8 },
  { id: '5', home: 'Tennessee', away: 'Auburn', homeML: -125, awayML: +105, totalLine: 141.5, modelPick: 'home', modelTotal: 138.9, modelHomeWin: 57.3, modelAwayWin: 42.7, vegasHomeWin: 55.6, vegasAwayWin: 44.4 },
];

const MOCK_GAMES_PAST = [
  { id: '1', home: 'UNC', away: 'Duke', homeML: -150, awayML: +130, totalLine: 145.5, modelPick: 'home', modelTotal: 148.2, modelHomeWin: 62.5, modelAwayWin: 37.5, vegasHomeWin: 58.0, vegasAwayWin: 42.0, actualTotal: 151, winner: 'home' as const },
  { id: '2', home: 'Kentucky', away: 'Kansas', homeML: +120, awayML: -140, totalLine: 152.0, modelPick: 'away', modelTotal: 155.8, modelHomeWin: 44.2, modelAwayWin: 55.8, vegasHomeWin: 46.5, vegasAwayWin: 53.5, actualTotal: 149, winner: 'home' as const },
  { id: '3', home: 'UCLA', away: 'Gonzaga', homeML: -110, awayML: -110, totalLine: 148.5, modelPick: 'home', modelTotal: 146.1, modelHomeWin: 51.8, modelAwayWin: 48.2, vegasHomeWin: 50.0, vegasAwayWin: 50.0, actualTotal: 144, winner: 'home' as const },
  { id: '4', home: 'Michigan St', away: 'Purdue', homeML: null, awayML: null, totalLine: 139.0, modelPick: 'away', modelTotal: 141.5, modelHomeWin: 38.9, modelAwayWin: 61.1, vegasHomeWin: 40.2, vegasAwayWin: 59.8, actualTotal: 145, winner: 'away' as const },
  { id: '5', home: 'Tennessee', away: 'Auburn', homeML: -125, awayML: +105, totalLine: 141.5, modelPick: 'home', modelTotal: 138.9, modelHomeWin: 57.3, modelAwayWin: 42.7, vegasHomeWin: 55.6, vegasAwayWin: 44.4, actualTotal: 136, winner: 'away' as const },
];

const formatOdds = (odds: number | null) => {
  if (odds === null) return '-';
  return odds > 0 ? `+${odds}` : `${odds}`;
};

interface GamesTableProps {
  isToday: boolean;
}

export default function GamesTable({ isToday }: GamesTableProps) {
  const games = isToday ? MOCK_GAMES_TODAY : MOCK_GAMES_PAST;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm font-medium border-collapse border border-slate-600">
        {/* Header */}
        <thead>
          <tr className="bg-slate-700 text-slate-200 font-semibold">
            <th className="px-3 py-1.5 border border-slate-600 text-left">MATCHUP</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">MODEL %</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">VEGAS %</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">TOTAL PRED</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">VEGAS O/U</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">HOME ML</th>
            <th className="px-3 py-1.5 border border-slate-600 text-center">AWAY ML</th>
          </tr>
        </thead>

        {/* Rows */}
        <tbody>
          {games.map((game, index) => {
            const pickedPct = game.modelPick === 'home' ? game.modelHomeWin : game.modelAwayWin;

            // For past days, determine if predictions were correct
            const isPastGame = 'winner' in game && 'actualTotal' in game;
            const pickCorrect = isPastGame ? game.modelPick === game.winner : null;
            const totalPredOver = game.modelTotal > game.totalLine;
            const actualWentOver = isPastGame ? game.actualTotal > game.totalLine : null;
            const totalCorrect = isPastGame ? totalPredOver === actualWentOver : null;

            return (
              <tr
                key={game.id}
                className={`${index % 2 === 0 ? 'bg-slate-900' : 'bg-slate-800'} hover:bg-slate-700 transition-colors`}
              >
                {/* Matchup */}
                <td className="px-3 py-1.5 border border-slate-700 whitespace-nowrap">
                  <span className={game.modelPick === 'away' ? 'text-orange-400 font-semibold' : 'text-slate-200'}>{game.away}</span>
                  <span className="text-slate-500 mx-1">@</span>
                  <span className={game.modelPick === 'home' ? 'text-orange-400 font-semibold' : 'text-slate-200'}>{game.home}</span>
                </td>

                {/* Model % */}
                <td className="px-3 py-1.5 border border-slate-700 text-center whitespace-nowrap">
                  <span className={`font-semibold ${
                    pickCorrect === true ? 'text-green-400' :
                    pickCorrect === false ? 'text-red-400' :
                    'text-orange-400'
                  }`}>{pickedPct.toFixed(0)}%</span>
                  <span className="text-slate-500">-</span>
                  <span className="text-slate-400">{(100 - pickedPct).toFixed(0)}%</span>
                </td>

                {/* Vegas % */}
                <td className="px-3 py-1.5 border border-slate-700 text-center text-slate-300 whitespace-nowrap">
                  {game.vegasHomeWin.toFixed(0)}%-{game.vegasAwayWin.toFixed(0)}%
                </td>

                {/* Total Pred */}
                <td className="px-3 py-1.5 border border-slate-700 text-center whitespace-nowrap">
                  <span className={game.modelTotal > game.totalLine ? 'text-green-400' : 'text-red-400'}>
                    {game.modelTotal.toFixed(1)}
                  </span>
                  <span className="text-slate-500 ml-1">({game.modelTotal > game.totalLine ? 'O' : 'U'})</span>
                </td>

                {/* Vegas O/U */}
                <td className="px-3 py-1.5 border border-slate-700 text-center text-slate-300">
                  {game.totalLine}
                </td>

                {/* Home ML */}
                <td className="px-3 py-1.5 border border-slate-700 text-center text-slate-300">
                  {formatOdds(game.homeML)}
                </td>

                {/* Away ML */}
                <td className="px-3 py-1.5 border border-slate-700 text-center text-slate-300">
                  {formatOdds(game.awayML)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
