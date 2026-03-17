import type { BracketGame } from '../../../api/ncaamb';

export const SLOT_H_BASE = 64;
export const CARD_H = 48;

export function slotHeight(roundIndex: number) {
  return SLOT_H_BASE * Math.pow(2, roundIndex);
}

interface Props {
  game: BracketGame;
  slotIndex: number;
  roundIndex: number;  // determines slot height: 0=smallest (R64), 3=largest (E8)
  isLastRound: boolean;
  mirrored?: boolean;  // connector on LEFT instead of right (for right-side bracket)
  compact?: boolean;   // narrower card width for desktop full-bracket
}

export default function BracketSlot({
  game, slotIndex, roundIndex, isLastRound, mirrored = false, compact = false,
}: Props) {
  const sH     = slotHeight(roundIndex);
  const padV   = (sH - CARD_H) / 2;
  const isUpper = slotIndex % 2 === 0;

  const t1Win = game.predicted_winner === game.pred_team_1;
  const t2Win = game.predicted_winner === game.pred_team_2;
  const hasResult = game.actual_winner !== null;
  const correct   = game.correct;
  const actualIsT1 = game.actual_winner === game.pred_team_1;
  const actualIsT2 = game.actual_winner === game.pred_team_2;

  const p1 = game.prob_ensemble ?? 0.5;
  const p2 = 1 - p1;

  const leftAccent = !hasResult
    ? 'border-l-2 border-l-slate-600'
    : correct === 1 ? 'border-l-2 border-l-green-500' : 'border-l-2 border-l-red-500';

  const cardW = compact ? 'w-32' : 'w-44';

  const card = (
    <div style={{ paddingTop: padV, paddingBottom: padV }} className={`${cardW} shrink-0`}>
      <div style={{ height: CARD_H }} className={`flex flex-col border border-slate-700 rounded overflow-hidden bg-slate-800 ${leftAccent}`}>
        {/* Team 1 */}
        <div className={`flex items-center gap-1 px-1.5 flex-1 border-b border-slate-700 ${t1Win ? 'bg-orange-500/20' : ''}`}>
          <span className="text-[10px] font-bold text-slate-500 w-4 text-center shrink-0">
            {game.pred_team_1_seed ?? '?'}
          </span>
          <span className={`text-[11px] truncate flex-1 leading-tight ${t1Win ? 'text-white font-semibold' : 'text-slate-300'}`}>
            {game.pred_team_1}
          </span>
          <span className={`text-[10px] font-mono shrink-0 ${t1Win ? 'text-orange-400' : 'text-slate-600'}`}>
            {(p1 * 100).toFixed(0)}%
          </span>
          {hasResult && actualIsT1 && (
            <span className="text-[10px] shrink-0 ml-0.5">{correct === 1 ? '✅' : '❌'}</span>
          )}
        </div>
        {/* Team 2 */}
        <div className={`flex items-center gap-1 px-1.5 flex-1 ${t2Win ? 'bg-orange-500/20' : ''}`}>
          <span className="text-[10px] font-bold text-slate-500 w-4 text-center shrink-0">
            {game.pred_team_2_seed ?? '?'}
          </span>
          <span className={`text-[11px] truncate flex-1 leading-tight ${t2Win ? 'text-white font-semibold' : 'text-slate-300'}`}>
            {game.pred_team_2}
          </span>
          <span className={`text-[10px] font-mono shrink-0 ${t2Win ? 'text-orange-400' : 'text-slate-600'}`}>
            {(p2 * 100).toFixed(0)}%
          </span>
          {hasResult && actualIsT2 && (
            <span className="text-[10px] shrink-0 ml-0.5">{correct === 1 ? '✅' : '❌'}</span>
          )}
        </div>
      </div>
    </div>
  );

  // Same connector CSS whether normal or mirrored —
  // flex-row-reverse positions the connector div on the left for mirrored.
  const connector = !isLastRound ? (
    <div className="w-5 shrink-0 flex flex-col">
      {isUpper ? (
        <>
          <div className="flex-1" />
          <div className="flex-1 border-r border-b border-slate-600" />
        </>
      ) : (
        <>
          <div className="flex-1 border-r border-t border-slate-600" />
          <div className="flex-1" />
        </>
      )}
    </div>
  ) : null;

  return (
    <div style={{ height: sH }} className={`flex ${mirrored ? 'flex-row-reverse' : ''}`}>
      {card}
      {connector}
    </div>
  );
}
