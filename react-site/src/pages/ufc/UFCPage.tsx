import { useState, useEffect } from 'react';

type Fight = {
  fight_id: string;
  event_id: string;
  fighter1_id: string;
  fighter2_id: string;
  fighter1_name: string;
  fighter2_name: string;
  fighter1_nickname: string | null;
  fighter2_nickname: string | null;
  fighter1_img_link: string | null;
  fighter2_img_link: string | null;
  algopick_model: string | null;
  algopick_prediction: number | null;
  algopick_probability: number | null;
  window_sample: number | null;
  correct: number | null;
  date: string;
  end_time: string | null;
  weight_class: string;
  win_method: string | null;
};

type BookmakerOdds = {
  fight_id: string;
  bookmaker: string;
  fighter1_id: string;
  fighter2_id: string;
  fighter1_odds: number;
  fighter2_odds: number;
  fighter1_odds_percent: number;
  fighter2_odds_percent: number;
  ev: number | null;
  vigor: number;
};

type Event = {
  event_id: string;
  event_url: string;
  title: string;
  event_datestr: string;
  location: string;
  date: string;
};

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const options: Intl.DateTimeFormatOptions = { 
    month: 'short', 
    day: 'numeric', 
    year: 'numeric' 
  };
  return date.toLocaleDateString('en-US', options);
}

function formatOdds(odds: number): string {
  return odds > 0 ? `+${odds}` : `${odds}`;
}

function getBookmakerDisplayName(key: string): string {
  const names: { [key: string]: string } = {
    'bovada': 'Bovada',
    'fanduel': 'FanDuel',
    'draftkings': 'DraftKings',
    'betmgm': 'BetMGM',
    'betonlineag': 'BetOnline',
    'betus': 'BetUS',
    'betrivers': 'BetRivers'
  };
  return names[key] || key;
}

export default function UFCPage() {
  const [activeTab, setActiveTab] = useState('upcoming');
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const [events, setEvents] = useState<{ upcoming: Event[]; past: Event[] }>({ 
    upcoming: [], 
    past: [] 
  });
  const [eventFights, setEventFights] = useState<{ [key: string]: Fight[] }>({});
  const [fightOdds, setFightOdds] = useState<{ [key: string]: BookmakerOdds[] }>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const upcomingRes = await fetch('http://localhost:5000/api/ufc/events/upcoming');
        const pastRes = await fetch('http://localhost:5000/api/ufc/events/past');
        
        const upcomingData = await upcomingRes.json();
        const pastData = await pastRes.json();
        
        setEvents({
          upcoming: upcomingData,
          past: pastData
        });
        setLoading(false);
      } catch (error) {
        console.error('Error fetching events:', error);
        setLoading(false);
      }
    };

    fetchEvents();
  }, []);

  const handleEventClick = async (eventId: string) => {
    if (expandedEvent === eventId) {
      setExpandedEvent(null);
      return;
    }

    setExpandedEvent(eventId);

    if (!eventFights[eventId]) {
      try {
        const response = await fetch(`http://localhost:5000/api/ufc/events/${eventId}/fights`);
        const fights = await response.json();
        
        const sortedFights = fights.sort((a: Fight, b: Fight) => {
          if (a.fight_type === 'title' && b.fight_type !== 'title') return -1;
          if (a.fight_type !== 'title' && b.fight_type === 'title') return 1;
          if (a.fight_type === 'main' && b.fight_type !== 'main') return -1;
          if (a.fight_type !== 'main' && b.fight_type === 'main') return 1;
          return 0;
        });
        
        setEventFights(prev => ({ ...prev, [eventId]: sortedFights }));

        // Fetch odds for all fights
        const oddsPromises = sortedFights.map((fight: Fight) =>
          fetch(`http://localhost:5000/api/ufc/fights/${fight.fight_id}/odds`)
            .then(res => res.json())
            .then(odds => ({ fightId: fight.fight_id, odds }))
            .catch(() => ({ fightId: fight.fight_id, odds: [] }))
        );

        const allOdds = await Promise.all(oddsPromises);
        const oddsMap: { [key: string]: BookmakerOdds[] } = {};
        allOdds.forEach(({ fightId, odds }) => {
          oddsMap[fightId] = odds;
        });
        setFightOdds(prev => ({ ...prev, ...oddsMap }));

      } catch (error) {
        console.error('Error fetching fights:', error);
      }
    }
  };

  const getBestEVOdds = (fight: Fight, odds: BookmakerOdds[]) => {
    if (!odds || odds.length === 0) return null;

    let bestFighter1: { bookmaker: string; odds: number; ev: number } | null = null;
    let bestFighter2: { bookmaker: string; odds: number; ev: number } | null = null;

    odds.forEach(bookmaker => {
      // For fighter1 - we want positive EV when fighter1 is predicted to win
      if (fight.algopick_prediction === 0 && bookmaker.ev !== null) {
        if (!bestFighter1 || bookmaker.ev > bestFighter1.ev) {
          bestFighter1 = {
            bookmaker: getBookmakerDisplayName(bookmaker.bookmaker),
            odds: bookmaker.fighter1_odds,
            ev: bookmaker.ev
          };
        }
      }
      
      // For fighter2 - we need to calculate EV based on their implied probability
      // EV for fighter2 = (fighter2_probability * fighter2_decimal_odds) - 1
      if (fight.algopick_prediction === 1) {
        const fighter2Prob = (100 - (fight.algopick_probability || 0)) / 100;
        const fighter2DecimalOdds = bookmaker.fighter2_odds > 0 
          ? (bookmaker.fighter2_odds / 100) + 1 
          : (100 / Math.abs(bookmaker.fighter2_odds)) + 1;
        const fighter2EV = (fighter2Prob * fighter2DecimalOdds) - 1;
        
        if (!bestFighter2 || fighter2EV > bestFighter2.ev) {
          bestFighter2 = {
            bookmaker: getBookmakerDisplayName(bookmaker.bookmaker),
            odds: bookmaker.fighter2_odds,
            ev: fighter2EV
          };
        }
      }
    });

    return { bestFighter1, bestFighter2 };
  };

  const currentEvents = events[activeTab as keyof typeof events];
  const isPastTab = activeTab === 'past';

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading events...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <nav className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="px-8">
          <div className="flex items-center justify-between h-16">
            <button 
              onClick={() => window.location.href = '/'}
              className="flex items-center hover:opacity-80 transition-opacity"
            >
              <span className="text-2xl font-bold text-white">
                Algo<span className="text-orange-500">Picks</span>
              </span>
            </button>
            <div className="flex items-center">
              <img src="/logo/ufc-logo.png" alt="UFC" className="h-8 object-contain" />
            </div>
          </div>
        </div>
      </nav>

      <div className="border-b border-slate-700 bg-slate-900/30 sticky top-16 z-40">
        <div className="px-8 flex items-center space-x-6">
          <button
            onClick={() => setActiveTab('upcoming')}
            className={`py-4 px-2 font-semibold transition-colors border-b-2 ${
              activeTab === 'upcoming'
                ? 'text-orange-500 border-orange-500'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            Upcoming
          </button>
          <button
            onClick={() => setActiveTab('past')}
            className={`py-4 px-2 font-semibold transition-colors border-b-2 ${
              activeTab === 'past'
                ? 'text-orange-500 border-orange-500'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            Past
          </button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
                        const bestOdds = hasPrediction ? getBestEVOdds(fight, odds) : null;

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
                        
                        const confidence = parseFloat(fight.algopick_probability);
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
                                {bestOdds?.bestFighter1 && fight.algopick_prediction === 0 && (
                                  <div className="text-right mt-1">
                                    <p className="text-green-400 font-semibold text-sm">
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
                                {bestOdds?.bestFighter2 && fight.algopick_prediction === 1 && (
                                  <div className="text-left mt-1">
                                    <p className="text-green-400 font-semibold text-sm">
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
                              <div className="mt-4 space-y-2">
                                <p className="text-sm font-semibold text-slate-300 mb-2">Bookmaker Odds:</p>
                                {odds.map((bookmaker) => {
                                  // Calculate EV for both fighters
                                  const fighter1Prob = (fight.algopick_prediction === 0 ? confidence : 100 - confidence) / 100;
                                  const fighter2Prob = (fight.algopick_prediction === 1 ? confidence : 100 - confidence) / 100;
                                  
                                  const fighter1DecimalOdds = bookmaker.fighter1_odds > 0 
                                    ? (bookmaker.fighter1_odds / 100) + 1 
                                    : (100 / Math.abs(bookmaker.fighter1_odds)) + 1;
                                  const fighter2DecimalOdds = bookmaker.fighter2_odds > 0 
                                    ? (bookmaker.fighter2_odds / 100) + 1 
                                    : (100 / Math.abs(bookmaker.fighter2_odds)) + 1;
                                  
                                  const fighter1EV = ((fighter1Prob * fighter1DecimalOdds) - 1) * 100;
                                  const fighter2EV = ((fighter2Prob * fighter2DecimalOdds) - 1) * 100;

                                  return (
                                    <div key={bookmaker.bookmaker} className="bg-slate-700/30 rounded-lg p-3">
                                      <div className="flex items-center justify-between mb-2">
                                        <p className="text-xs font-semibold text-slate-300">
                                          {getBookmakerDisplayName(bookmaker.bookmaker)}
                                        </p>
                                      </div>
                                      
                                      {/* Fighter labels */}
                                      <div className="flex justify-between text-[10px] text-slate-400 mb-1 px-1">
                                        <span>{fight.fighter1_name}</span>
                                        <span>{fight.fighter2_name}</span>
                                      </div>

                                      <div className="relative h-8 bg-slate-900 rounded overflow-hidden mb-2">
                                        {/* Fighter 1 section */}
                                        <div 
                                          className="absolute top-0 left-0 h-full bg-blue-500"
                                          style={{ width: `${bookmaker.fighter1_odds_percent}%` }}
                                        ></div>
                                        
                                        {/* Vigor section */}
                                        <div 
                                          className="absolute top-0 h-full bg-yellow-500"
                                          style={{ 
                                            left: `${bookmaker.fighter1_odds_percent}%`,
                                            width: `${bookmaker.vigor}%`
                                          }}
                                        ></div>
                                        
                                        {/* Fighter 2 section */}
                                        <div 
                                          className="absolute top-0 right-0 h-full bg-red-500"
                                          style={{ width: `${bookmaker.fighter2_odds_percent}%` }}
                                        ></div>
                                        
                                        {/* Text overlay */}
                                        <div className="absolute inset-0 flex items-center justify-between px-3">
                                          <span className="text-sm font-bold text-white drop-shadow-lg">
                                            {formatOdds(bookmaker.fighter1_odds)}
                                          </span>
                                          <span className="text-[10px] font-bold text-slate-900 bg-yellow-300 px-2 py-0.5 rounded">
                                            Vig: {bookmaker.vigor.toFixed(1)}%
                                          </span>
                                          <span className="text-sm font-bold text-white drop-shadow-lg">
                                            {formatOdds(bookmaker.fighter2_odds)}
                                          </span>
                                        </div>
                                      </div>
                                      
                                      {/* EV display */}
                                      <div className="flex justify-between text-xs font-semibold">
                                        <div className="flex items-center space-x-1">
                                          <span className={fighter1EV > 0 ? 'text-green-400' : 'text-red-400'}>
                                            EV: {fighter1EV > 0 ? '+' : ''}{fighter1EV.toFixed(1)}%
                                          </span>
                                          {fighter1EV > 5 && (
                                            <span className="text-[10px] bg-green-500 text-white px-2 py-0.5 rounded font-bold">
                                              +EV BET
                                            </span>
                                          )}
                                        </div>
                                        <div className="flex items-center space-x-1">
                                          <span className={fighter2EV > 0 ? 'text-green-400' : 'text-red-400'}>
                                            EV: {fighter2EV > 0 ? '+' : ''}{fighter2EV.toFixed(1)}%
                                          </span>
                                          {fighter2EV > 5 && (
                                            <span className="text-[10px] bg-green-500 text-white px-2 py-0.5 rounded font-bold">
                                              +EV BET
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  );
                                })}
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
      </div>
    </div>
  );
}