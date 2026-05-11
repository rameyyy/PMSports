import type { Fight } from '../types';
import { formatOdds, capitalizeName, formatMethod } from '../utils';

interface Props {
  fight: Fight;
  isPast: boolean;
}

export default function FightCard({ fight, isPast }: Props) {
  const f1Prob = fight.f1_probability != null ? parseFloat(String(fight.f1_probability)) * 100 : null;
  const f2Prob = f1Prob != null ? 100 - f1Prob : null;
  const hasPrediction = fight.predicted_winner_id != null && f1Prob != null;

  const pickedF1 = fight.predicted_winner_id === fight.fighter1_id;
  const pickedF2 = fight.predicted_winner_id === fight.fighter2_id;

  // Vegas pick derived from odds
  const hasOdds = fight.f1_odds != null && fight.f2_odds != null;
  const vegasPickF1 = hasOdds && fight.f1_odds! < fight.f2_odds!;
  const vegasPickF2 = hasOdds && fight.f2_odds! < fight.f1_odds!;
  const vegasDisagrees = hasPrediction && hasOdds && ((pickedF1 && vegasPickF2) || (pickedF2 && vegasPickF1));

  // Draw / NC detection
  const isDraw = fight.actual_winner_id === 'drawornc' || fight.actual_winner_id === 'draw';

  // Settled = model has a result OR it's a draw/NC
  const showResult = isPast && (fight.correct !== null || isDraw);
  const isCorrect = fight.correct === 1;

  // Who actually won — derived from actual_winner_id for the "WINNER" badge
  const actualWinnerIsF1 = showResult && !isDraw && !isCorrect && pickedF2;
  const actualWinnerIsF2 = showResult && !isDraw && !isCorrect && pickedF1;

  const imgClass = (picked: boolean, isActualWinner: boolean) => {
    if (isDraw) return 'border-2 border-slate-500';
    if (showResult) {
      if (picked && isCorrect) return 'border-[3px] border-green-500';
      if (picked && !isCorrect) return 'border-[3px] border-red-500';
      if (isActualWinner) return 'border-[3px] border-green-500 opacity-100';
    }
    if (picked) return 'border-[3px] border-orange-500';
    return 'border-2 border-slate-600';
  };

  const nameClass = (picked: boolean) => {
    if (!picked) return 'text-white';
    if (!showResult || isDraw) return 'text-orange-400';
    return isCorrect ? 'text-green-400' : 'text-red-400';
  };

  const f1Name = capitalizeName(fight.fighter1_name);
  const f2Name = capitalizeName(fight.fighter2_name);

  return (
    <div className="px-6 sm:px-10 py-5 border-b border-slate-700/50 last:border-0">
      {/* Header row */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-semibold text-orange-500 uppercase tracking-wider">
          {fight.weight_class}
        </span>
        <div className="flex items-center gap-2">
          {fight.fight_type === 'title' && (
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
              TITLE FIGHT
            </span>
          )}
          {fight.fight_type === 'main' && (
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
              MAIN EVENT
            </span>
          )}
          {isDraw && (
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-500/30 text-slate-300 border border-slate-500/40">
              DRAW / NC
            </span>
          )}
          {showResult && !isDraw && (
            <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
              isCorrect ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {isCorrect ? 'Correct' : 'Incorrect'}
            </span>
          )}
          {!isPast && vegasDisagrees && (
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
              SPLIT PICK
            </span>
          )}
        </div>
      </div>

      {/* Fighters row */}
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4 sm:gap-8">
        {/* Fighter 1 */}
        <div className="flex items-center gap-4">
          {fight.fighter1_img_link && (
            <img
              src={fight.fighter1_img_link}
              alt={f1Name}
              className={`w-14 h-14 sm:w-16 sm:h-16 rounded-full object-cover flex-shrink-0 ${imgClass(pickedF1, actualWinnerIsF1)}`}
              onError={(e) => (e.currentTarget.style.display = 'none')}
            />
          )}
          <div>
            <p className={`font-bold text-sm sm:text-lg leading-tight ${nameClass(pickedF1)}`}>{f1Name}</p>
            {fight.fighter1_nickname && (
              <p className="text-slate-500 text-xs italic">"{capitalizeName(fight.fighter1_nickname)}"</p>
            )}
            {fight.f1_odds != null && (
              <p className="text-slate-400 text-xs mt-0.5">{formatOdds(fight.f1_odds)}</p>
            )}
            <div className="flex gap-1.5 mt-1 flex-wrap">
              {pickedF1 && (
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                  showResult && !isDraw
                    ? isCorrect ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    : 'bg-orange-500/20 text-orange-400'
                }`}>
                  ALGO PICK
                </span>
              )}
              {vegasPickF1 && (
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                  !isPast || !showResult || isDraw
                    ? vegasDisagrees ? 'bg-purple-500/20 text-purple-400' : 'bg-slate-600/40 text-slate-400'
                    : fight.actual_winner_id === fight.fighter1_id
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-red-500/20 text-red-400'
                }`}>
                  VEGAS
                </span>
              )}
              {actualWinnerIsF1 && (
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                  WINNER
                </span>
              )}
              {showResult && !isDraw && isCorrect && pickedF1 && (
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                  WINNER
                </span>
              )}
            </div>
          </div>
        </div>

        <span className="text-slate-500 font-semibold text-sm text-center">VS</span>

        {/* Fighter 2 — mirrored */}
        <div className="flex items-center gap-4 justify-end">
          <div className="text-right">
            <p className={`font-bold text-sm sm:text-lg leading-tight ${nameClass(pickedF2)}`}>{f2Name}</p>
            {fight.fighter2_nickname && (
              <p className="text-slate-500 text-xs italic">"{capitalizeName(fight.fighter2_nickname)}"</p>
            )}
            {fight.f2_odds != null && (
              <p className="text-slate-400 text-xs mt-0.5">{formatOdds(fight.f2_odds)}</p>
            )}
            <div className="flex gap-1.5 mt-1 flex-wrap justify-end">
              {pickedF2 && (
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                  showResult && !isDraw
                    ? isCorrect ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    : 'bg-orange-500/20 text-orange-400'
                }`}>
                  ALGO PICK
                </span>
              )}
              {vegasPickF2 && (
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                  !isPast || !showResult || isDraw
                    ? vegasDisagrees ? 'bg-purple-500/20 text-purple-400' : 'bg-slate-600/40 text-slate-400'
                    : fight.actual_winner_id === fight.fighter2_id
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-red-500/20 text-red-400'
                }`}>
                  VEGAS
                </span>
              )}
              {actualWinnerIsF2 && (
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                  WINNER
                </span>
              )}
              {showResult && !isDraw && isCorrect && pickedF2 && (
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                  WINNER
                </span>
              )}
            </div>
          </div>
          {fight.fighter2_img_link && (
            <img
              src={fight.fighter2_img_link}
              alt={f2Name}
              className={`w-14 h-14 sm:w-16 sm:h-16 rounded-full object-cover flex-shrink-0 ${imgClass(pickedF2, actualWinnerIsF2)}`}
              onError={(e) => (e.currentTarget.style.display = 'none')}
            />
          )}
        </div>
      </div>

      {/* Probability bar */}
      {hasPrediction && (
        <div className="mt-4">
          <div className="relative h-5 bg-slate-700/50 rounded-full overflow-hidden">
            <div
              className={`absolute top-0 left-0 h-full ${pickedF1 ? 'bg-gradient-to-r from-orange-500 to-orange-400' : 'bg-slate-600/60'}`}
              style={{ width: `${f1Prob}%` }}
            />
            <div
              className={`absolute top-0 right-0 h-full ${pickedF2 ? 'bg-gradient-to-l from-orange-500 to-orange-400' : 'bg-slate-600/60'}`}
              style={{ width: `${f2Prob}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-between px-3 text-xs font-semibold text-white">
              <span>{f1Prob!.toFixed(1)}%</span>
              <span>{f2Prob!.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {!hasPrediction && (
        <p className="text-slate-500 text-xs mt-2">No prediction available</p>
      )}

      {/* Result line for past fights */}
      {showResult && !isDraw && (() => {
        const winnerName = isCorrect
          ? (pickedF1 ? f1Name : f2Name)
          : (pickedF1 ? f2Name : f1Name);
        const method = formatMethod(fight.win_method);
        const isDecision = fight.win_method?.startsWith('d_');
        const timePart = !isDecision && fight.end_time ? ` · ${fight.end_time}` : '';
        return (
          <p className="mt-2 text-xs text-slate-400">
            {winnerName} wins{method ? ` · ${method}` : ''}{timePart}
          </p>
        );
      })()}

      {isDraw && (
        <p className="mt-2 text-xs text-slate-400">
          {formatMethod(fight.win_method) || 'Draw / No Contest'}
          {fight.end_time ? ` · ${fight.end_time}` : ''}
        </p>
      )}
    </div>
  );
}
