import BracketSlot from './BracketSlot';
import type { BracketGame } from '../../../api/ncaamb';

const FORWARD_ROUNDS  = ['First Round', 'Second Round', 'Sweet 16', 'Elite 8'];
const MIRRORED_ROUNDS = ['Elite 8', 'Sweet 16', 'Second Round', 'First Round'];

const ROUND_LABELS: Record<string, string> = {
  'First Round':  'Round of 64',
  'Second Round': 'Round of 32',
  'Sweet 16':     'Sweet 16',
  'Elite 8':      'Elite 8',
};

// Fixed header height — imported by BracketDesktop for CenterColumn math
export const ROUND_HDR_H = 28;

interface Props {
  games: BracketGame[];
  mirrored?: boolean;
  compact?: boolean;
}

export default function BracketTree({ games, mirrored = false, compact = false }: Props) {
  const roundOrder = mirrored ? MIRRORED_ROUNDS : FORWARD_ROUNDS;
  const numRounds  = roundOrder.length;

  const columns = roundOrder
    .map((round, di) => ({
      round,
      displayIndex: di,
      roundIndex: mirrored ? (numRounds - 1 - di) : di,
      games: games.filter(g => g.round === round),
    }))
    .filter(col => col.games.length > 0);

  if (columns.length === 0) return null;

  const cardW = compact ? 'w-32' : 'w-44';

  return (
    <div className="flex">
      {columns.map((col, ci) => {
        // For mirrored, E8 is displayed first (ci=0) but should have no connector —
        // an explicit horizontal connector is drawn in BracketDesktop instead.
        const isLastRound = ci === columns.length - 1 || (mirrored && ci === 0);
        return (
          <div key={col.round} className="shrink-0">
            {/* Round header — fixed height for alignment with CenterColumn */}
            <div
              className={`${cardW} flex items-center justify-center px-1`}
              style={{ height: ROUND_HDR_H }}
            >
              <span className="text-xs font-semibold text-white tracking-wide">
                {ROUND_LABELS[col.round] ?? col.round}
              </span>
            </div>
            {/* Game slots */}
            {col.games.map((game, gi) => (
              <BracketSlot
                key={game.bracket_slot}
                game={game}
                slotIndex={gi}
                roundIndex={col.roundIndex}
                isLastRound={isLastRound}
                mirrored={mirrored}
                compact={compact}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}
