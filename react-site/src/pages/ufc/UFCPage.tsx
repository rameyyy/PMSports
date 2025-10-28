import { useState, useEffect, Suspense, lazy } from 'react';
import type { Fight, BookmakerOdds, Event } from './types';
import {
  formatDate,
  formatOdds,
  getBookmakerDisplayName
} from './utils';
import {
  fetchUpcomingEvents,
  fetchPastEvents,
  fetchEventFights,
  fetchMultipleFightOdds
} from './api';

const Models = lazy(() => import('./Models'));
const Bets = lazy(() => import('./Bets'));

export default function UFCPage() {
  const [activeTab, setActiveTab] = useState('upcoming');
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const [events, setEvents] = useState<{ upcoming: Event[]; past: Event[] }>({
    upcoming: [],
    past: []
  });
  const [eventFights, setEventFights] = useState<{ [key: string]: Fight[] }>({});
  const [fightOdds, setFightOdds] = useState<{ [key: string]: BookmakerOdds[] }>({});
  const [expandedOdds, setExpandedOdds] = useState<{ [key: string]: boolean }>({});
  const [initialLoading, setInitialLoading] = useState(true);
  const [pastPage, setPastPage] = useState(1);
  const [pendingPastPage, setPendingPastPage] = useState(1);
  const [pastLoading, setPastLoading] = useState(false);
  const [pastPagination, setPastPagination] = useState<{ total: number; total_pages: number; limit: number } | null>(null);
  const eventsPerPage = 10;

  // Fetch upcoming events once on mount
  useEffect(() => {
    const loadUpcomingEvents = async () => {
      try {
        const upcomingData = await fetchUpcomingEvents();
        setEvents(prev => ({ ...prev, upcoming: upcomingData }));
      } catch (error) {
        console.error('Error fetching upcoming events:', error);
      }
    };

    loadUpcomingEvents();
  }, []);

  // Fetch past events when pending page changes
  useEffect(() => {
    const loadPastEvents = async () => {
      setPastLoading(true);
      try {
        const pastData = await fetchPastEvents(pendingPastPage, eventsPerPage);
        setEvents(prev => ({ ...prev, past: pastData.events }));
        setPastPagination(pastData.pagination);
        // Only update the displayed page number AFTER data loads
        setPastPage(pendingPastPage);
      } catch (error) {
        console.error('Error fetching past events:', error);
      } finally {
        setPastLoading(false);
        setInitialLoading(false);
      }
    };

    loadPastEvents();
  }, [pendingPastPage]);

  const handleEventClick = async (eventId: string) => {
    if (expandedEvent === eventId) {
      setExpandedEvent(null);
      return;
    }

    setExpandedEvent(eventId);

    if (!eventFights[eventId]) {
      try {
        const fights = await fetchEventFights(eventId);

        const sortedFights = fights.sort((a: Fight, b: Fight) => {
          if (a.fight_type === 'title' && b.fight_type !== 'title') return -1;
          if (a.fight_type !== 'title' && b.fight_type === 'title') return 1;
          if (a.fight_type === 'main' && b.fight_type !== 'main') return -1;
          if (a.fight_type !== 'main' && b.fight_type === 'main') return 1;
          return 0;
        });

        setEventFights(prev => ({ ...prev, [eventId]: sortedFights }));

        // Fetch odds for all fights
        const oddsMap = await fetchMultipleFightOdds(sortedFights);
        setFightOdds(prev => ({ ...prev, ...oddsMap }));

      } catch (error) {
        console.error('Error fetching fights:', error);
      }
    }
  };
  const getBestEVOdds = (odds: BookmakerOdds[]): { bestFighter1: { bookmaker: string; odds: number; ev: number } | null; bestFighter2: { bookmaker: string; odds: number; ev: number } | null } | null => {
    if (!odds || odds.length === 0) return null;

    let bestFighter1: { bookmaker: string; odds: number; ev: number } | null = null;
    let bestFighter2: { bookmaker: string; odds: number; ev: number } | null = null;

    odds.forEach(bookmaker => {
      // For fighter1 - use backend calculated EV
      if (bookmaker.fighter1_ev !== null) {
        if (!bestFighter1 || bookmaker.fighter1_ev > bestFighter1.ev) {
          bestFighter1 = {
            bookmaker: getBookmakerDisplayName(bookmaker.bookmaker),
            odds: bookmaker.fighter1_odds,
            ev: bookmaker.fighter1_ev
          };
        }
      }
      
      // For fighter2 - use backend calculated EV
      if (bookmaker.fighter2_ev !== null) {
        if (!bestFighter2 || bookmaker.fighter2_ev > bestFighter2.ev) {
          bestFighter2 = {
            bookmaker: getBookmakerDisplayName(bookmaker.bookmaker),
            odds: bookmaker.fighter2_odds,
            ev: bookmaker.fighter2_ev
          };
        }
      }
    });

    return { bestFighter1, bestFighter2 };
  };

  const toggleOdds = (fightId: string) => {
    setExpandedOdds(prev => ({
      ...prev,
      [fightId]: !prev[fightId]
    }));
  };

  const currentEvents = events[activeTab as keyof typeof events];
  const isPastTab = activeTab === 'past';

  if (initialLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading events...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <nav className="border-b border-slate-700 bg-slate-900 sticky top-0 z-50">
        <div className="px-4 sm:px-8">
          <div className="flex items-center justify-between h-16">
            <button
              onClick={() => window.location.href = '/'}
              className="flex items-center hover:opacity-80 transition-opacity"
            >
              <span className="text-lg sm:text-2xl font-bold text-white">
                Algo<span className="text-orange-500">Picks</span>
              </span>
            </button>
            <div className="flex items-center">
              <img src="/logo/ufc-logo.png" alt="UFC" className="h-6 sm:h-8 object-contain" />
            </div>
          </div>
        </div>
      </nav>

      <div className="border-b border-slate-700 bg-slate-800 sticky top-16 z-40">
        <div className="px-4 sm:px-8 flex items-center space-x-3 sm:space-x-6 overflow-x-auto">
          <button
            onClick={() => setActiveTab('upcoming')}
            className={`py-4 px-2 sm:px-3 text-sm sm:text-base font-semibold transition-colors border-b-2 whitespace-nowrap ${
              activeTab === 'upcoming'
                ? 'text-orange-500 border-orange-500 bg-slate-700/20'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            Upcoming
          </button>
          <button
            onClick={() => setActiveTab('past')}
            className={`py-4 px-2 sm:px-3 text-sm sm:text-base font-semibold transition-colors border-b-2 whitespace-nowrap ${
              activeTab === 'past'
                ? 'text-orange-500 border-orange-500 bg-slate-700/20'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            Past
          </button>
          <button
            onClick={() => setActiveTab('models')}
            className={`py-4 px-2 sm:px-3 text-sm sm:text-base font-semibold transition-colors border-b-2 whitespace-nowrap ${
              activeTab === 'models'
                ? 'text-orange-500 border-orange-500 bg-slate-700/20'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            Models
          </button>
          <button
            onClick={() => setActiveTab('bets')}
            className={`py-4 px-2 sm:px-3 text-sm sm:text-base font-semibold transition-colors border-b-2 whitespace-nowrap ${
              activeTab === 'bets'
                ? 'text-orange-500 border-orange-500 bg-slate-700/20'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            Bets
          </button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Models Tab - Keep mounted but hidden for caching */}
        <div className={activeTab === 'models' ? '' : 'hidden'}>
          <Suspense fallback={
            <div className="text-center text-white py-12">
              Loading Models...
            </div>
          }>
            <Models />
          </Suspense>
        </div>

        {/* Bets Tab - Keep mounted but hidden for caching */}
        <div className={activeTab === 'bets' ? '' : 'hidden'}>
          <Suspense fallback={
            <div className="text-center text-white py-12">
              Loading Bets...
            </div>
          }>
            <Bets />
          </Suspense>
        </div>

        {/* Upcoming/Past Events Tabs */}
        {activeTab !== 'models' && activeTab !== 'bets' && (
          <>
            {currentEvents.length === 0 ? (
              <div className="text-center text-slate-400 py-12">
                No {activeTab} events found
              </div>
            ) : (
              <div className="space-y-4">
            {currentEvents.map((event) => (
              <div
                key={event.event_id}
                className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden"
              >
                <button
                  onClick={() => handleEventClick(event.event_id)}
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-800/80 transition-colors"
                >
                  <div className="text-left">
                    <h3 className="text-xl font-bold text-white mb-1">{event.title}</h3>
                    <p className="text-slate-200 text-base">{formatDate(event.date)}</p>
                    <p className="text-slate-200 text-sm">{event.location}</p>
                  </div>
                  <svg
                    className={`w-6 h-6 text-slate-400 transition-transform ${
                      expandedEvent === event.event_id ? 'rotate-180' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {expandedEvent === event.event_id && (
                  <div className="border-t border-slate-700">
                    {!eventFights[event.event_id] ? (
                      <div className="px-6 py-8 text-center text-slate-400">
                        Loading fights...
                      </div>
                    ) : eventFights[event.event_id].length === 0 ? (
                      <div className="px-6 py-8 text-center text-slate-400">
                        No fights available for this event
                      </div>
                    ) : (
                      eventFights[event.event_id].map((fight) => {
                        const hasPrediction = fight.algopick_prediction !== null && 
                                            fight.algopick_prediction !== undefined &&
                                            fight.algopick_probability !== null;
                        
                        const odds = fightOdds[fight.fight_id] || [];
                        const bestOdds = hasPrediction ? getBestEVOdds(odds) : null;

                        if (!hasPrediction) {
                          return (
                            <div key={fight.fight_id} className="px-6 py-4 border-b border-slate-700/50 last:border-0">
                              <div className="grid grid-cols-3 gap-4 items-center mb-2">
                                <p className="text-white font-semibold text-right">{fight.fighter1_name}</p>
                                <span className="text-slate-500 font-bold text-center">VS</span>
                                <p className="text-white font-semibold text-left">{fight.fighter2_name}</p>
                              </div>
                              <div className="text-center text-slate-500 text-sm mt-2">
                                No prediction available
                              </div>
                            </div>
                          );
                        }

                        const predictedWinnerName = fight.algopick_prediction === 0 
                          ? fight.fighter1_name 
                          : fight.fighter2_name;
                        
                        const confidence = fight.algopick_probability == null ? 0: parseFloat(String(fight.algopick_probability));
                        const predictedProb = confidence;
                        const underDogProb = 100 - confidence;
                        
                        const showResult = isPastTab && fight.correct !== null;
                        const predictionCorrect = fight.correct === 1;

                        return (
                          <div key={fight.fight_id} className="px-6 py-4 border-b border-slate-700/50 last:border-0">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-xs font-semibold text-orange-500 uppercase tracking-wide">
                                  {fight.weight_class}
                                </span>
                              </div>
                              <div className="flex items-center space-x-2">
                                {fight.fight_type === 'title' && (
                                  <span className="text-xs font-semibold px-3 py-1 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                                    TITLE FIGHT
                                  </span>
                                )}
                                {fight.fight_type === 'main' && (
                                  <span className="text-xs font-semibold px-3 py-1 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                                    MAIN EVENT
                                  </span>
                                )}
                              </div>
                              <div>
                                {showResult && (
                                  <span
                                    className={`text-sm font-semibold px-3 py-1.5 rounded ${
                                      predictionCorrect
                                        ? 'bg-green-500/20 text-green-400'
                                        : 'bg-red-500/20 text-red-400'
                                    }`}
                                  >
                                    {predictionCorrect ? 'Correct' : 'Incorrect'}
                                  </span>
                                )}
                              </div>
                            </div>

                            <div className="grid grid-cols-3 gap-4 items-center mb-4">
                              <div className="flex flex-col items-end">
                                {fight.fighter1_img_link && (
                                  <img 
                                    src={fight.fighter1_img_link} 
                                    alt={fight.fighter1_name}
                                    className={`w-20 h-20 rounded-full object-cover mb-2 ${
                                      fight.algopick_prediction === 0 && !showResult
                                        ? 'border-4 border-orange-500 shadow-lg shadow-orange-500/50'
                                        : fight.algopick_prediction === 0 && showResult && predictionCorrect
                                        ? 'border-4 border-green-500 shadow-lg shadow-green-500/50'
                                        : fight.algopick_prediction === 0 && showResult && !predictionCorrect
                                        ? 'border-4 border-red-500 shadow-lg shadow-red-500/50'
                                        : 'border-2 border-slate-600'
                                    }`}
                                    onError={(e) => e.currentTarget.style.display = 'none'}
                                  />
                                )}
                                <p className={`font-semibold text-xl text-right ${
                                  fight.algopick_prediction === 0 ? 'text-orange-400' : 'text-white'
                                }`}>
                                  {fight.fighter1_name}
                                </p>
                                {fight.fighter1_nickname && (
                                  <p className="text-slate-200 text-base italic text-right">
                                    "{fight.fighter1_nickname}"
                                  </p>
                                )}
                                {bestOdds?.bestFighter1 && (
                                  <div className="text-right mt-1">
                                    <p className="text-orange-300 font-semibold text-sm">
                                      {formatOdds(bestOdds.bestFighter1.odds)}
                                    </p>
                                  </div>
                                )}
                              </div>
                              
                              <div className="text-center">
                                <span className="text-white font-bold text-xl">VS</span>
                              </div>
                              
                              <div className="flex flex-col items-start">
                                {fight.fighter2_img_link && (
                                  <img 
                                    src={fight.fighter2_img_link} 
                                    alt={fight.fighter2_name}
                                    className={`w-20 h-20 rounded-full object-cover mb-2 ${
                                      fight.algopick_prediction === 1 && !showResult
                                        ? 'border-4 border-orange-500 shadow-lg shadow-orange-500/50'
                                        : fight.algopick_prediction === 1 && showResult && predictionCorrect
                                        ? 'border-4 border-green-500 shadow-lg shadow-green-500/50'
                                        : fight.algopick_prediction === 1 && showResult && !predictionCorrect
                                        ? 'border-4 border-red-500 shadow-lg shadow-red-500/50'
                                        : 'border-2 border-slate-600'
                                    }`}
                                    onError={(e) => e.currentTarget.style.display = 'none'}
                                  />
                                )}
                                <p className={`font-semibold text-xl text-left ${
                                  fight.algopick_prediction === 1 ? 'text-orange-400' : 'text-white'
                                }`}>
                                  {fight.fighter2_name}
                                </p>
                                {fight.fighter2_nickname && (
                                  <p className="text-slate-200 text-base italic text-left">
                                    "{fight.fighter2_nickname}"
                                  </p>
                                )}
                                {bestOdds?.bestFighter2 && (
                                  <div className="text-left mt-1">
                                    <p className="text-orange-300 font-semibold text-sm">
                                      {formatOdds(bestOdds.bestFighter2.odds)}
                                    </p>
                                  </div>
                                )}
                              </div>
                            </div>

                            {showResult && fight.win_method && (
                              <div className="mb-3">
                                <div className="text-sm mb-1">
                                  <span className="text-slate-200 font-bold">Winner: </span>
                                  <span className="text-white font-semibold text-lg">
                                    {predictionCorrect ? predictedWinnerName : (fight.algopick_prediction === 0 ? fight.fighter2_name : fight.fighter1_name)}
                                  </span>
                                </div>
                                <div className="text-xs">
                                  <span className="text-slate-200 font-bold">AlgoPick: </span>
                                  <span className="font-bold text-base text-orange-400">
                                    {predictedWinnerName}
                                  </span>
                                  <span className="text-white ml-2 text-sm font-semibold">
                                    ({confidence.toFixed(1)}%)
                                  </span>
                                </div>
                              </div>
                            )}
                            
                            {!showResult && (
                              <div className="text-sm mb-3">
                                <span className="text-slate-200 font-bold">AlgoPick: </span>
                                <span className="font-bold text-xl text-orange-400">
                                  {predictedWinnerName}
                                </span>
                                <span className="text-white ml-2 text-lg font-semibold">
                                  ({confidence.toFixed(1)}%)
                                </span>
                              </div>
                            )}

                            <div className="mt-4">
                              <div className="relative h-8 bg-slate-700/50 rounded-lg overflow-hidden">
                                <div 
                                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-orange-500 to-orange-600 transition-all"
                                  style={{ width: `${predictedProb}%` }}
                                ></div>
                                <div 
                                  className="absolute top-0 right-0 h-full bg-gradient-to-r from-slate-500 to-slate-600 transition-all"
                                  style={{ width: `${underDogProb}%` }}
                                ></div>
                                <div className="absolute inset-0 flex items-center justify-between px-3 text-sm font-semibold text-white">
                                  <span>{predictedProb.toFixed(1)}%</span>
                                  <span>{underDogProb.toFixed(1)}%</span>
                                </div>
                              </div>
                            </div>

                            {/* Bookmaker Odds Section */}
                            {odds.length > 0 && (
                              <div className="mt-4">
                                <button
                                  onClick={() => toggleOdds(fight.fight_id)}
                                  className="w-full flex items-center justify-between px-4 py-3 bg-slate-700/50 hover:bg-slate-700/70 rounded-lg transition-colors"
                                >
                                  <span className="text-base font-semibold text-white">
                                    View Bookmaker Odds ({odds.length})
                                  </span>
                                  <svg
                                    className={`w-5 h-5 text-slate-400 transition-transform ${
                                      expandedOdds[fight.fight_id] ? 'rotate-180' : ''
                                    }`}
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                  </svg>
                                </button>

                                {expandedOdds[fight.fight_id] && (
                                  <div className="mt-2 space-y-2">
                                    {odds.map((bookmaker) => {
                                    // Use backend-calculated EV values
                                    const fighter1EV = bookmaker.fighter1_ev ?? 0;
                                    const fighter2EV = bookmaker.fighter2_ev ?? 0;

                                    return (
                                      <div key={bookmaker.bookmaker} className="bg-slate-700/30 rounded-lg p-3">
                                        <div className="flex items-center justify-between mb-2">
                                          <p className="text-xs font-semibold text-slate-300">
                                            {getBookmakerDisplayName(bookmaker.bookmaker)}
                                          </p>
                                        </div>
                                        
                                        {/* Fighter labels */}
                                        <div className="flex justify-between text-sm font-semibold mb-1 px-1">
                                          <span className={fight.algopick_prediction === 0 ? 'text-orange-400' : 'text-white'}>
                                            {fight.fighter1_name}
                                          </span>
                                          <span className={fight.algopick_prediction === 1 ? 'text-orange-400' : 'text-white'}>
                                            {fight.fighter2_name}
                                          </span>
                                        </div>

                                        <div className="relative h-8 bg-slate-900 rounded overflow-hidden mb-2">
                                          {/* Fighter 1 section */}
                                          <div 
                                            className="absolute top-0 left-0 h-full bg-orange-700"
                                            style={{ width: `${bookmaker.fighter1_odds_percent}%` }}
                                          ></div>
                                          
                                          {/* Vigor section */}
                                          <div 
                                            className="absolute top-0 h-full bg-yellow-400"
                                            style={{ 
                                              left: `${bookmaker.fighter1_odds_percent}%`,
                                              width: `${bookmaker.vigor}%`
                                            }}
                                          ></div>
                                          
                                          {/* Fighter 2 section */}
                                          <div 
                                            className="absolute top-0 right-0 h-full bg-gray-500"
                                            style={{ width: `${bookmaker.fighter2_odds_percent}%` }}
                                          ></div>
                                          
                                          {/* Text overlay */}
                                          <div className="absolute inset-0 flex items-center justify-between px-3">
                                            <span className="text-xs font-bold text-white drop-shadow-lg">
                                              {formatOdds(bookmaker.fighter1_odds)}
                                            </span>
                                            <span className="text-[10px] font-bold text-slate-900 bg-yellow-300 px-2 py-0.5 rounded">
                                              Vig: {bookmaker.vigor.toFixed(1)}%
                                            </span>
                                            <span className="text-xs font-bold text-white drop-shadow-lg">
                                              {formatOdds(bookmaker.fighter2_odds)}
                                            </span>
                                          </div>
                                        </div>
                                        
                                        {/* EV display */}
                                        <div className="flex justify-between text-xs font-semibold">
                                          <div className="flex items-center space-x-1">
                                            <span className={fighter1EV > 0 ? 'text-green-400' : 'text-red-400'}>
                                              <span className="text-white">{formatOdds(bookmaker.fighter1_odds)} as Percent: {bookmaker.fighter1_odds_percent}%</span>
                                              <br />EV: {fighter1EV > 0 ? '+' : ''}{fighter1EV.toFixed(1)}%<br />
                                            </span>
                                          </div>
                                          <div className="flex items-center justify-end space-x-1">
                                            <div className="text-right">
                                              <div className="text-white">{formatOdds(bookmaker.fighter2_odds)} as Percent: {bookmaker.fighter2_odds_percent}%</div>
                                              <div className={fighter2EV > 0 ? 'text-green-400' : 'text-red-400'}>
                                                EV: {fighter2EV > 0 ? '+' : ''}{fighter2EV.toFixed(1)}%
                                              </div>
                                            </div>
                                          </div>
                                        </div>
                                      </div>
                                    );
                                  })}
                                  </div>
                                )}
                              </div>
                            )}

                            {showResult && fight.win_method && (
                              <div className="mt-3 text-sm">
                                <span className="text-slate-400">Finish: </span>
                                <span className="text-slate-200 font-medium">{fight.win_method}</span>
                                {fight.end_time && <span className="text-slate-400"> ({fight.end_time})</span>}
                              </div>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Pagination controls for past events */}
        {activeTab === 'past' && pastPagination && pastPagination.total_pages > 1 && (
          <div className="flex items-center justify-center gap-2 sm:gap-4 mt-8">
            <button
              onClick={() => setPendingPastPage(p => Math.max(1, p - 1))}
              disabled={pastPage === 1 || pastLoading}
              className="px-3 sm:px-4 py-2 text-sm sm:text-base bg-slate-700 text-white rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            <span className="text-xs sm:text-base text-slate-300 flex items-center gap-2">
              Page {pastPage} of {pastPagination.total_pages}
              {pastLoading && (
                <svg className="w-4 h-4 animate-spin text-orange-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              )}
            </span>
            <button
              onClick={() => setPendingPastPage(p => Math.min(pastPagination.total_pages, p + 1))}
              disabled={pastPage === pastPagination.total_pages || pastLoading}
              className="px-3 sm:px-4 py-2 text-sm sm:text-base bg-slate-700 text-white rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}