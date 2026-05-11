import type { Event, Fight } from '../types';
import { formatDate } from '../utils';
import FightCard from './FightCard';

interface Props {
  event: Event;
  expanded: boolean;
  onToggle: () => void;
  fights: Fight[] | null;
  isPast: boolean;
}

function computeEventStats(fights: Fight[]) {
  let modelW = 0, modelL = 0;
  let vegasW = 0, vegasL = 0;
  let disagreeModelW = 0, disagreeVegasW = 0, disagreeBothW = 0;

  for (const f of fights) {
    const isDraw = f.actual_winner_id === 'drawornc' || f.actual_winner_id === 'draw';
    if (isDraw || f.correct === null || f.actual_winner_id === null) continue;

    const pickedF1 = f.predicted_winner_id === f.fighter1_id;
    const modelRight = f.correct === 1;
    if (modelRight) modelW++; else modelL++;

    const hasOdds = f.f1_odds != null && f.f2_odds != null;
    if (!hasOdds) continue;

    const vegasFavorF1 = f.f1_odds! < f.f2_odds!;
    const vegasFavorF2 = f.f2_odds! < f.f1_odds!;
    if (!vegasFavorF1 && !vegasFavorF2) continue; // pick 'em, skip

    const vegasPickId = vegasFavorF1 ? f.fighter1_id : f.fighter2_id;
    const vegasRight = vegasPickId === f.actual_winner_id;
    if (vegasRight) vegasW++; else vegasL++;

    const modelPickId = pickedF1 ? f.fighter1_id : f.fighter2_id;
    const disagrees = modelPickId !== vegasPickId;
    if (disagrees) {
      if (modelRight && !vegasRight) disagreeModelW++;
      else if (!modelRight && vegasRight) disagreeVegasW++;
      else disagreeBothW++;
    }
  }

  return { modelW, modelL, vegasW, vegasL, disagreeModelW, disagreeVegasW, disagreeBothW };
}

export default function EventCard({ event, expanded, onToggle, fights, isPast }: Props) {
  const stats = isPast && fights && fights.length > 0 ? computeEventStats(fights) : null;
  const disagreeTotal = stats ? stats.disagreeModelW + stats.disagreeVegasW + stats.disagreeBothW : 0;

  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-800/80 transition-colors text-left"
      >
        <div>
          <h3 className="text-base sm:text-xl font-bold text-white">{event.title}</h3>
          <p className="text-slate-400 text-sm mt-0.5">{formatDate(event.date)} · {event.location}</p>
        </div>
        <svg
          className={`w-5 h-5 text-slate-400 flex-shrink-0 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="border-t border-slate-700">
          {fights === null ? (
            <p className="py-8 text-center text-slate-400">Loading fights...</p>
          ) : fights.length === 0 ? (
            <p className="py-8 text-center text-slate-400">No fights available</p>
          ) : (
            <>
              {isPast && stats && (
                <div className="px-6 sm:px-10 py-3 bg-slate-900/40 border-b border-slate-700/50 flex flex-wrap gap-x-6 gap-y-1 text-xs">
                  <span className="text-slate-400">
                    Model:{' '}
                    <span className="text-green-400 font-semibold">{stats.modelW}W</span>
                    {' '}<span className="text-red-400 font-semibold">{stats.modelL}L</span>
                  </span>
                  <span className="text-slate-400">
                    Vegas:{' '}
                    <span className="text-green-400 font-semibold">{stats.vegasW}W</span>
                    {' '}<span className="text-red-400 font-semibold">{stats.vegasL}L</span>
                  </span>
                  {disagreeTotal > 0 && (
                    <span className="text-slate-400">
                      Disagreements ({disagreeTotal}):{' '}
                      <span className="text-orange-400 font-semibold">Model {stats.disagreeModelW}</span>
                      {' / '}
                      <span className="text-blue-400 font-semibold">Vegas {stats.disagreeVegasW}</span>
                      {stats.disagreeBothW > 0 && (
                        <span className="text-slate-500"> / Both wrong {stats.disagreeBothW}</span>
                      )}
                    </span>
                  )}
                </div>
              )}
              {fights.map(fight => (
                <FightCard key={fight.fight_id} fight={fight} isPast={isPast} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
