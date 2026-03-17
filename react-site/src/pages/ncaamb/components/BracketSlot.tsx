import type { BracketGame } from '../../../api/ncaamb';

export const SLOT_H_BASE = 64;
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

  const t1IsPick   = game.predicted_winner === game.pred_team_1;
  const t2IsPick   = game.predicted_winner === game.pred_team_2;
  const hasResult  = game.actual_winner !== null;
  const correct    = game.correct;
  const t1Won      = hasResult && game.actual_winner === game.pred_team_1;
  const t2Won      = hasResult && game.actual_winner === game.pred_team_2;

  const p1 = game.prob_ensemble ?? 0.5;
  const p2 = 1 - p1;

  // Per-row background
  function rowBg(isPick: boolean, isWon: boolean) {
    if (!hasResult) return isPick ? 'bg-orange-500/20' : '';
    if (isPick && correct === 1) return 'bg-green-500/25';
    if (isPick && correct === 0) return 'bg-red-500/20';
    if (isWon)                   return 'bg-green-500/25';
    return '';
  }

  // Team name color
  function nameColor(isPick: boolean, isWon: boolean) {
    if (!hasResult) return isPick ? 'text-white font-semibold' : 'text-slate-200';
    if (isPick && correct === 1) return 'text-green-300 font-semibold';
    if (isPick && correct === 0) return 'text-red-300 font-semibold';
    if (isWon)                   return 'text-green-300 font-semibold';
    return 'text-slate-300';
  }

  // Small badge shown after team name
  function badge(isPick: boolean, isWon: boolean) {
    if (!hasResult) {
      if (isPick) return <span className="text-[8px] font-bold uppercase tracking-wide text-orange-400 shrink-0">PICK</span>;
      return null;
    }
    if (isPick && correct === 1)
      return <span className="text-[8px] font-bold uppercase tracking-wide text-green-400 shrink-0">PICK</span>;
    if (isPick && correct === 0)
      return <span className="text-[8px] font-bold uppercase tracking-wide text-red-400 shrink-0">PICK</span>;
    if (isWon)
      return <span className="text-[8px] font-bold uppercase tracking-wide text-green-400 shrink-0">WON</span>;
    return null;
  }

  // Overall left accent
  const leftAccent = !hasResult
    ? 'border-l-2 border-l-slate-600'
    : correct === 1
      ? 'border-l-2 border-l-green-500'
      : 'border-l-2 border-l-red-500';

  const cardW = compact ? 'w-32' : 'w-44';

  const card = (
    <div style={{ paddingTop: padV, paddingBottom: padV }} className={`${cardW} shrink-0`}>
      <div style={{ height: cardH }} className={`flex flex-col border border-slate-700 rounded overflow-hidden bg-slate-800 ${leftAccent}`}>
        {/* Team 1 */}
        <div className={`flex items-center gap-1 px-1.5 flex-1 border-b border-slate-700 ${rowBg(t1IsPick, t1Won)}`}>
          <span className="text-[10px] font-bold text-slate-300 w-4 text-center shrink-0">
            {game.pred_team_1_seed ?? '?'}
          </span>
          <span className={`text-[11px] truncate flex-1 leading-tight ${nameColor(t1IsPick, t1Won)}`}>
            {game.pred_team_1}
          </span>
          {badge(t1IsPick, t1Won)}
          <span className="text-[10px] font-mono text-slate-300 shrink-0 ml-0.5">
            {(p1 * 100).toFixed(0)}%
          </span>
        </div>
        {/* Team 2 */}
        <div className={`flex items-center gap-1 px-1.5 flex-1 ${rowBg(t2IsPick, t2Won)}`}>
          <span className="text-[10px] font-bold text-slate-300 w-4 text-center shrink-0">
            {game.pred_team_2_seed ?? '?'}
          </span>
          <span className={`text-[11px] truncate flex-1 leading-tight ${nameColor(t2IsPick, t2Won)}`}>
            {game.pred_team_2}
          </span>
          {badge(t2IsPick, t2Won)}
          <span className="text-[10px] font-mono text-slate-300 shrink-0 ml-0.5">
            {(p2 * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );

  return (
    <div style={{ height: sH }} className="flex">
      {card}
    </div>
  );
}
