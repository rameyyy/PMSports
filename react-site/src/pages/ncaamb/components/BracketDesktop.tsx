import { useRef, useCallback } from 'react';
import BracketSlot, { SLOT_H_BASE, CARD_H } from './BracketSlot';
import BracketTree, { ROUND_HDR_H } from './BracketTree';
import type { BracketGame } from '../../../api/ncaamb';

// ── Layout constants ──────────────────────────────────────────────────────────
const REGION_H       = 8 * SLOT_H_BASE;   // 512px — 8 R64 game slots
const REGION_LABEL_H = 24;                 // "East"/"South" label row (fixed)
const DIVIDER_H      = 24;                 // my-3 between the two regions (12+12px margin)

// Total height of one region block (label + tree header + game slots)
const REGION_BLOCK_H = REGION_LABEL_H + ROUND_HDR_H + REGION_H;   // 564px
// Total height of both regions + divider
const TOTAL_H        = REGION_BLOCK_H * 2 + DIVIDER_H;             // 1152px

// Center card positions — where each card's vertical center should land
const FF1_CENTER   = REGION_LABEL_H + ROUND_HDR_H + REGION_H / 2;
const FF2_CENTER   = REGION_BLOCK_H + DIVIDER_H + REGION_LABEL_H + ROUND_HDR_H + REGION_H / 2;
const CHAMP_CENTER = (FF1_CENTER + FF2_CENTER) / 2;

// Each absolute wrapper contains: label (CENTER_LABEL_H) + BracketSlot with roundIndex=0
// BracketSlot roundIndex=0: slotHeight=SLOT_H_BASE, padV=(SLOT_H_BASE-CARD_H)/2
// So card center is CENTER_LABEL_H + padV + CARD_H/2 below the wrapper top
const CENTER_LABEL_H  = 16;                              // text-[10px] + mb-1 ≈ 16px
const CENTER_PAD_V    = (SLOT_H_BASE - CARD_H) / 2;     // 8px
const CENTER_CARD_OFF = CENTER_LABEL_H + CENTER_PAD_V + CARD_H / 2; // 40px

const FF1_TOP   = FF1_CENTER   - CENTER_CARD_OFF;
const CHAMP_TOP = CHAMP_CENTER - CENTER_CARD_OFF;
const FF2_TOP   = FF2_CENTER   - CENTER_CARD_OFF;

// ── DragScroll ────────────────────────────────────────────────────────────────
function DragScroll({ children }: { children: React.ReactNode }) {
  const ref  = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; sl: number; st: number } | null>(null);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    const el = ref.current;
    if (!el) return;
    drag.current = { x: e.clientX, y: e.clientY, sl: el.scrollLeft, st: el.scrollTop };
    el.style.cursor = 'grabbing';
    e.preventDefault();
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!drag.current || !ref.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    ref.current.scrollLeft = drag.current.sl - dx;
    ref.current.scrollTop  = drag.current.st - dy;
  }, []);

  const stopDrag = useCallback(() => {
    if (ref.current) ref.current.style.cursor = 'grab';
    drag.current = null;
  }, []);

  return (
    <div
      ref={ref}
      className="overflow-scroll pb-4 select-none no-scrollbar"
      style={{ cursor: 'grab', maxHeight: '80vh' }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={stopDrag}
      onMouseLeave={stopDrag}
    >
      {children}
    </div>
  );
}

// ── E8↔FF horizontal connector column ────────────────────────────────────────
// Renders two horizontal tick lines aligned with FF_G1 and FF_G2 card centers
function FFConnector() {
  return (
    <div className="relative shrink-0" style={{ width: 14, height: TOTAL_H }}>
      <div className="absolute w-full border-t border-slate-500" style={{ top: FF1_CENTER }} />
      <div className="absolute w-full border-t border-slate-500" style={{ top: FF2_CENTER }} />
    </div>
  );
}

// ── CenterColumn ──────────────────────────────────────────────────────────────
interface CenterProps { games: BracketGame[] }

function CenterColumn({ games }: CenterProps) {
  const ffG1  = games.find(g => g.bracket_slot === 'FF_G1');
  const ffG2  = games.find(g => g.bracket_slot === 'FF_G2');
  const champ = games.find(g => g.round === 'Championship');

  return (
    <div className="shrink-0 flex flex-col items-center" style={{ width: 148 }}>
      <div className="relative w-full" style={{ height: TOTAL_H }}>
        {/* Vertical line connecting FF_G1 → Championship → FF_G2 */}
        <div
          className="absolute border-l border-slate-500"
          style={{ left: '50%', top: FF1_CENTER, height: FF2_CENTER - FF1_CENTER }}
        />
        {ffG1 && (
          <div className="absolute w-full px-1" style={{ top: FF1_TOP }}>
            <div className="text-[10px] font-semibold text-slate-400 text-center mb-1">Final Four</div>
            <BracketSlot game={ffG1} slotIndex={0} roundIndex={0} isLastRound compact />
          </div>
        )}
        {champ && (
          <div className="absolute w-full px-1" style={{ top: CHAMP_TOP }}>
            <div className="text-[10px] font-semibold text-orange-400 text-center mb-1">Championship</div>
            <BracketSlot game={champ} slotIndex={0} roundIndex={0} isLastRound compact />
          </div>
        )}
        {ffG2 && (
          <div className="absolute w-full px-1" style={{ top: FF2_TOP }}>
            <div className="text-[10px] font-semibold text-slate-400 text-center mb-1">Final Four</div>
            <BracketSlot game={ffG2} slotIndex={0} roundIndex={0} isLastRound compact />
          </div>
        )}
      </div>
    </div>
  );
}

// ── BracketDesktop ────────────────────────────────────────────────────────────
interface Props { games: BracketGame[] }

export default function BracketDesktop({ games }: Props) {
  const firstFour  = games.filter(g => g.round === 'First Four');
  const ffAndChamp = games.filter(g => g.round === 'Final Four' || g.round === 'Championship');

  const regionGames = (r: string) =>
    games.filter(g => g.region === r && g.round !== 'First Four');

  return (
    <div>
      {/* First Four strip */}
      {firstFour.length > 0 && (
        <div className="mb-5">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            First Four Play-In Games
          </p>
          <div className="flex flex-wrap gap-3">
            {firstFour.map(g => (
              <BracketSlot key={g.bracket_slot} game={g} slotIndex={0} roundIndex={0} isLastRound compact />
            ))}
          </div>
        </div>
      )}

      {/* Full bracket — drag to pan */}
      <DragScroll>
        <div className="min-w-full flex justify-center py-2">
        <div className="flex w-fit">
          {/* Left: East (top) + South (bottom) */}
          <div className="shrink-0 flex flex-col">
            <div
              className="text-sm font-extrabold text-white uppercase tracking-widest pl-1 flex items-center"
              style={{ height: REGION_LABEL_H }}
            >
              East
            </div>
            <BracketTree games={regionGames('East')} compact />
            <div className="border-t border-slate-700/50" style={{ marginTop: DIVIDER_H / 2, marginBottom: DIVIDER_H / 2 }} />
            <div
              className="text-sm font-extrabold text-white uppercase tracking-widest pl-1 flex items-center"
              style={{ height: REGION_LABEL_H }}
            >
              South
            </div>
            <BracketTree games={regionGames('South')} compact />
          </div>

          {/* E8 → FF connector */}
          <FFConnector />

          {/* Center: Final Four + Championship */}
          <CenterColumn games={ffAndChamp} />

          {/* FF ← E8 connector (right side) */}
          <FFConnector />

          {/* Right: West (top, mirrored) + Midwest (bottom, mirrored) */}
          <div className="shrink-0 flex flex-col">
            <div
              className="text-sm font-extrabold text-white uppercase tracking-widest pl-1 flex items-center"
              style={{ height: REGION_LABEL_H }}
            >
              West
            </div>
            <BracketTree games={regionGames('West')} mirrored compact />
            <div className="border-t border-slate-700/50" style={{ marginTop: DIVIDER_H / 2, marginBottom: DIVIDER_H / 2 }} />
            <div
              className="text-sm font-extrabold text-white uppercase tracking-widest pl-1 flex items-center"
              style={{ height: REGION_LABEL_H }}
            >
              Midwest
            </div>
            <BracketTree games={regionGames('Midwest')} mirrored compact />
          </div>
        </div>
        </div>
      </DragScroll>
    </div>
  );
}
