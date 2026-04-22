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

export default function EventCard({ event, expanded, onToggle, fights, isPast }: Props) {
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
            fights.map(fight => (
              <FightCard key={fight.fight_id} fight={fight} isPast={isPast} />
            ))
          )}
        </div>
      )}
    </div>
  );
}
