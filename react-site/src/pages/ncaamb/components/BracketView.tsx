import { useState, useEffect } from 'react';
import { fetchBracket, type BracketGame } from '../../../api/ncaamb';
import BracketSlot from './BracketSlot';
import BracketTree from './BracketTree';
import BracketDesktop from './BracketDesktop';

type Tab = 'East' | 'South' | 'West' | 'Midwest' | 'Final Four';

const TABS: { id: Tab; label: string }[] = [
  { id: 'East',       label: 'East' },
  { id: 'South',      label: 'South' },
  { id: 'West',       label: 'West' },
  { id: 'Midwest',    label: 'Midwest' },
  { id: 'Final Four', label: '🏆 Final Four' },
];

function RecordPill({ games }: { games: BracketGame[] }) {
  const played  = games.filter(g => g.correct !== null).length;
  const correct = games.filter(g => g.correct === 1).length;
  if (played === 0) return null;
  return (
    <span className="ml-1.5 text-[11px] font-mono opacity-80">
      {correct}/{played}
    </span>
  );
}

function FinalFourMobile({ games }: { games: BracketGame[] }) {
  const semis = games.filter(g => g.round === 'Final Four');
  const champ = games.find(g => g.round === 'Championship');
  return (
    <div className="overflow-x-auto pb-2">
      <div className="flex w-fit">
        <div className="shrink-0">
          <div className="text-sm font-semibold text-white mb-2 w-44 text-center">Final Four</div>
          {semis.map((g, i) => (
            <BracketSlot key={g.bracket_slot} game={g} slotIndex={i} roundIndex={0} isLastRound={false} />
          ))}
        </div>
        {champ && (
          <div className="shrink-0">
            <div className="text-sm font-semibold text-orange-400 mb-2 w-44 text-center">🏆 Championship</div>
            <BracketSlot game={champ} slotIndex={0} roundIndex={1} isLastRound />
          </div>
        )}
      </div>
    </div>
  );
}

// ── Mobile view ───────────────────────────────────────────────────────────────
function MobileView({ games }: { games: BracketGame[] }) {
  const [activeTab, setActiveTab] = useState<Tab>('East');

  const firstFour = games.filter(g => g.round === 'First Four');
  const tabGames = (tab: Tab): BracketGame[] => {
    if (tab === 'Final Four')
      return games.filter(g => g.round === 'Final Four' || g.round === 'Championship');
    return games.filter(g => g.region === tab);
  };
  const treeGames = (region: Exclude<Tab, 'Final Four'>) =>
    games.filter(g => g.region === region && g.round !== 'First Four');

  return (
    <div>
      {/* First Four strip */}
      {firstFour.length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">First Four</p>
          <div className="flex flex-wrap gap-2">
            {firstFour.map(g => (
              <BracketSlot key={g.bracket_slot} game={g} slotIndex={0} roundIndex={0} isLastRound />
            ))}
          </div>
        </div>
      )}

      {/* Region tabs — pill style */}
      <div className="flex gap-2 overflow-x-auto pb-2 mb-4 scrollbar-hide">
        {TABS.map(tab => {
          const tg = tabGames(tab.id);
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1 px-4 py-2 rounded-full text-sm font-semibold whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-orange-500 text-white shadow-lg shadow-orange-500/25 scale-105'
                  : 'bg-slate-700/80 text-slate-300 hover:bg-slate-600 hover:text-white border border-slate-600'
              }`}
            >
              {tab.label}
              <RecordPill games={tg} />
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="overflow-x-auto pb-2">
        {activeTab === 'Final Four' ? (
          <FinalFourMobile games={tabGames('Final Four')} />
        ) : (
          <BracketTree games={treeGames(activeTab)} />
        )}
      </div>
    </div>
  );
}

// ── Root component ────────────────────────────────────────────────────────────
export default function BracketView() {
  const [games, setGames]     = useState<BracketGame[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBracket()
      .then(d => setGames(d.games))
      .catch(() => setGames([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-center py-20 text-slate-400">Loading bracket...</div>;
  if (!games.length) return <div className="text-center py-20 text-slate-400">No bracket data.</div>;

  const allPlayed  = games.filter(g => g.correct !== null).length;
  const allCorrect = games.filter(g => g.correct === 1).length;

  return (
    <div>
      {allPlayed > 0 && (
        <div className="mb-4 flex items-center gap-2">
          <span className="text-sm text-slate-400">Record:</span>
          <span className={`text-sm font-semibold ${allCorrect / allPlayed >= 0.6 ? 'text-green-400' : 'text-orange-400'}`}>
            {allCorrect}/{allPlayed} ({Math.round(allCorrect / allPlayed * 100)}%)
          </span>
        </div>
      )}

      {/* Mobile: tabs per region */}
      <div className="lg:hidden">
        <MobileView games={games} />
      </div>

      {/* Desktop: full bracket */}
      <div className="hidden lg:block">
        <BracketDesktop games={games} />
      </div>
    </div>
  );
}
