import { useState, useEffect } from 'react';
import type { Event, Fight } from './types';
import { fetchUpcomingEvents, fetchPastEvents, fetchEventFights } from './api';
import type { PaginationInfo } from './api';
import EventCard from './components/EventCard';

type Tab = 'upcoming' | 'past';

export default function UFCPage() {
  const [activeTab, setActiveTab] = useState<Tab>('upcoming');
  const [upcomingEvents, setUpcomingEvents] = useState<Event[]>([]);
  const [pastEvents, setPastEvents] = useState<Event[]>([]);
  const [eventFights, setEventFights] = useState<Record<string, Fight[]>>({});
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [pastLoading, setPastLoading] = useState(false);
  const [pastPage, setPastPage] = useState(1);
  const [pendingPastPage, setPendingPastPage] = useState(1);
  const [pastPagination, setPastPagination] = useState<PaginationInfo | null>(null);

  useEffect(() => {
    fetchUpcomingEvents().then(setUpcomingEvents).catch(console.error);
  }, []);

  useEffect(() => {
    setPastLoading(true);
    fetchPastEvents(pendingPastPage, 10)
      .then(data => {
        setPastEvents(data.events);
        setPastPagination(data.pagination);
        setPastPage(pendingPastPage);
      })
      .catch(console.error)
      .finally(() => { setPastLoading(false); setLoading(false); });
  }, [pendingPastPage]);

  const handleEventToggle = async (eventId: string) => {
    if (expandedEvent === eventId) { setExpandedEvent(null); return; }
    setExpandedEvent(eventId);
    if (!eventFights[eventId]) {
      try {
        const fights = await fetchEventFights(eventId);
        const sorted = [...fights].sort((a, b) => {
          if (a.fight_type === 'title') return -1;
          if (b.fight_type === 'title') return 1;
          if (a.fight_type === 'main') return -1;
          if (b.fight_type === 'main') return 1;
          return 0;
        });
        setEventFights(prev => ({ ...prev, [eventId]: sorted }));
      } catch (e) {
        console.error(e);
      }
    }
  };

  const currentEvents = activeTab === 'upcoming' ? upcomingEvents : pastEvents;

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <p className="text-white text-xl">Loading events...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <nav className="border-b border-slate-700 bg-slate-900 sticky top-0 z-50">
        <div className="px-4 sm:px-8 flex items-center justify-between h-16">
          <button onClick={() => window.location.href = '/'} className="hover:opacity-80 transition-opacity">
            <span className="text-lg sm:text-2xl font-bold text-white">
              Algo<span className="text-orange-500">Picks</span>
            </span>
          </button>
          <img src="/logo/ufc-logo.png" alt="UFC" className="h-6 sm:h-8 object-contain" />
        </div>
      </nav>

      <div className="border-b border-slate-700 bg-slate-800 sticky top-16 z-40">
        <div className="px-4 sm:px-8 flex gap-6">
          {(['upcoming', 'past'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-4 px-1 text-sm font-semibold border-b-2 capitalize transition-colors ${
                activeTab === tab
                  ? 'text-orange-500 border-orange-500'
                  : 'text-slate-400 border-transparent hover:text-slate-300'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentEvents.length === 0 ? (
          <p className="text-center text-slate-400 py-12">No {activeTab} events found</p>
        ) : (
          <div className="space-y-3">
            {currentEvents.map(event => (
              <EventCard
                key={event.event_id}
                event={event}
                expanded={expandedEvent === event.event_id}
                onToggle={() => handleEventToggle(event.event_id)}
                fights={eventFights[event.event_id] ?? null}
                isPast={activeTab === 'past'}
              />
            ))}
          </div>
        )}

        {activeTab === 'past' && pastPagination && pastPagination.total_pages > 1 && (
          <div className="flex items-center justify-center gap-4 mt-8">
            <button
              onClick={() => setPendingPastPage(p => Math.max(1, p - 1))}
              disabled={pastPage === 1 || pastLoading}
              className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 disabled:opacity-50 transition-colors text-sm"
            >
              Previous
            </button>
            <span className="text-slate-300 text-sm flex items-center gap-2">
              Page {pastPage} of {pastPagination.total_pages}
              {pastLoading && (
                <svg className="w-4 h-4 animate-spin text-orange-500" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              )}
            </span>
            <button
              onClick={() => setPendingPastPage(p => Math.min(pastPagination.total_pages, p + 1))}
              disabled={pastPage === pastPagination.total_pages || pastLoading}
              className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 disabled:opacity-50 transition-colors text-sm"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
