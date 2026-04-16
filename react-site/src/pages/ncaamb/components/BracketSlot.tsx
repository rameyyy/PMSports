import type { BracketGame } from '../../../api/ncaamb';

export const SLOT_H_BASE = 96;
export const CARD_H = 48;

export function slotHeight(roundIndex: number) {
  return SLOT_H_BASE * Math.pow(2, roundIndex);
}

interface Props {
  game: BracketGame;
  slotIndex: number;
  roundIndex: number;
  isLastRound?: boolean;
  mirrored?: boolean;
  compact?: boolean;
  cardH?: number;  // override default CARD_H
}

export default function BracketSlot({
  game, roundIndex, compact = false, cardH = CARD_H,
}: Props) {
  const sH   = slotHeight(roundIndex);
  const padV = (sH - cardH) / 2;

  const hasResult = game.actual_winner !== null;
  const correct = game.correct;

  // Use actual teams if available, otherwise use predictions
  const team1 = hasResult && game.actual_team_1 ? game.actual_team_1 : game.pred_team_1;
  const team2 = hasResult && game.actual_team_2 ? game.actual_team_2 : game.pred_team_2;

  // Determine if each team won
  const t1Won = hasResult && game.actual_winner === team1;
  const t2Won = hasResult && game.actual_winner === team2;

  // Check if predicted winner was in this game
  const pickWasInGame = game.predicted_winner === team1 || game.predicted_winner === team2;

  const p1 = game.prob_ensemble ?? 0.5;
  const p2 = 1 - p1;

  // Per-row background - highlight winner in green
  function rowBg(isWon: boolean) {
    if (!hasResult) return '';
    if (isWon) return 'bg-green-500/25';
    return '';
  }

  // Team name color
  function nameColor(isWon: boolean) {
    if (!hasResult) return 'text-slate-200';
    if (isWon) return 'text-green-300 font-semibold';
    return 'text-slate-300';
  }

  // Pick indicator badge
  function pickBadge() {
    if (!hasResult) {
      return (
        <div className="text-[9px] text-slate-400 mt-0.5">
          Pick: <span className="text-orange-400 font-semibold">{game.predicted_winner}</span>
        </div>
      );
    }

    if (correct === 1) {
      return (
        <div className="text-[9px] text-slate-400 mt-0.5">
          Pick: <span className="text-green-400 font-semibold">{game.predicted_winner}</span> ✓
        </div>
      );
    }

    if (correct === 0 && pickWasInGame) {
      return (
        <div className="text-[9px] text-slate-400 mt-0.5">
          Pick: <span className="text-red-400 font-semibold">{game.predicted_winner}</span> ✗
        </div>
      );
    }

    if (correct === 0 && !pickWasInGame) {
      return (
        <div className="text-[9px] text-slate-400 mt-0.5">
          Pick: <span className="text-slate-500 font-semibold">{game.predicted_winner}</span> (out)
        </div>
      );
    }

    return null;
  }

  // Overall left accent
  const leftAccent = !hasResult
    ? 'border-l-2 border-l-slate-600'
    : correct === 1
      ? 'border-l-2 border-l-green-500'
      : correct === 0 && pickWasInGame
        ? 'border-l-2 border-l-red-500'
        : 'border-l-2 border-l-slate-500';

  const cardW = compact ? 'w-32' : 'w-44';

  const card = (
    <div style={{ paddingTop: padV, paddingBottom: padV }} className={`${cardW} shrink-0`}>
      <div className={`border border-slate-700 rounded overflow-hidden bg-slate-800 ${leftAccent}`}>
        <div style={{ minHeight: cardH }} className="flex flex-col">
          {/* Team 1 */}
          <div className={`flex items-center gap-1 px-1.5 py-1 border-b border-slate-700 ${rowBg(t1Won)}`}>
            <span className="text-[10px] font-bold text-slate-300 w-4 text-center shrink-0">
              {game.pred_team_1_seed ?? '?'}
            </span>
            <span className={`text-[11px] truncate flex-1 leading-tight ${nameColor(t1Won)}`}>
              {team1}
            </span>
            {t1Won && <span className="text-[8px] font-bold uppercase tracking-wide text-green-400 shrink-0">WON</span>}
            <span className="text-[10px] font-mono text-slate-300 shrink-0 ml-0.5">
              {(p1 * 100).toFixed(0)}%
            </span>
          </div>
          {/* Team 2 */}
          <div className={`flex items-center gap-1 px-1.5 py-1 ${rowBg(t2Won)}`}>
            <span className="text-[10px] font-bold text-slate-300 w-4 text-center shrink-0">
              {game.pred_team_2_seed ?? '?'}
            </span>
            <span className={`text-[11px] truncate flex-1 leading-tight ${nameColor(t2Won)}`}>
              {team2}
            </span>
            {t2Won && <span className="text-[8px] font-bold uppercase tracking-wide text-green-400 shrink-0">WON</span>}
            <span className="text-[10px] font-mono text-slate-300 shrink-0 ml-0.5">
              {(p2 * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        {/* Pick indicator */}
        {hasResult && (
          <div className="px-1.5 py-1 border-t border-slate-700/50 bg-slate-900/50">
            {pickBadge()}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div style={{ height: sH }} className="flex">
      {card}
    </div>
  );
}
