import { useState, useEffect } from 'react';
import Navbar from './components/home/Navbar';
import SportNavBar from './components/home/SportNavBar';
import SportSection from './components/home/SportSection';
import { fetchHomepageStats, fetchPickOfDay } from './api/ncaamb';

export default function HomePage() {
  const [basketballStats, setBasketballStats] = useState({
    gamesCount: 0,
    modelAccuracy: 0,
    modelCorrect: 0,
    modelTotal: 0,
    vegasAccuracy: 0,
    vegasCorrect: 0,
    vegasTotal: 0,
    edge: 0
  });
  const [pickOfDay, setPickOfDay] = useState({
    todayPick: null as any,
    yesterdayPick: null as any,
    record: { correct: 0, total: 0, accuracy: 0 }
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchHomepageStats(), fetchPickOfDay()])
      .then(([stats, pickData]) => {
        setBasketballStats({
          gamesCount: stats.todays_games,
          modelAccuracy: stats.model_accuracy,
          modelCorrect: stats.model_correct,
          modelTotal: stats.model_total,
          vegasAccuracy: stats.vegas_accuracy,
          vegasCorrect: stats.vegas_correct,
          vegasTotal: stats.vegas_total,
          edge: stats.edge
        });
        setPickOfDay({
          todayPick: pickData.today_pick,
          yesterdayPick: pickData.yesterday_pick,
          record: pickData.record
        });
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch data:', err);
        setLoading(false);
      });
  }, []);

  // Sport navigation data
  const sports = [
    { name: "NCAA Men's Basketball", logo: "/logo/ncaa-logo.png", path: "/cbb", available: false },
    { name: "MMA", logo: "/logo/ufc-logo.png", path: "/ufc", available: true }
  ];

  const formatOdds = (odds: number) => {
    return odds > 0 ? `+${odds}` : `${odds}`;
  };

  const basketballData = {
    name: "NCAA Basketball",
    subtitle: "Daily games (Nov - Apr)",
    logo: "/logo/ncaa-logo.png",
    path: "/cbb",
    available: false,
    nextEvent: {
      name: basketballStats.gamesCount === -1 ? "Games updating..." : "Today's Games",
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      gamesCount: basketballStats.gamesCount === -1 ? 0 : basketballStats.gamesCount
    },
    modelAccuracy: {
      winRate: loading ? "-.-%%" : `${basketballStats.modelAccuracy.toFixed(1)}%`,
      record: loading ? "-/-" : `${basketballStats.modelCorrect}-${basketballStats.modelTotal}`,
      totalPicks: basketballStats.modelTotal
    },
    vegasAccuracy: {
      winRate: loading ? "-.-%%" : `${basketballStats.vegasAccuracy.toFixed(1)}%`,
      record: loading ? "-/-" : `${basketballStats.vegasCorrect}-${basketballStats.vegasTotal}`,
      totalPicks: basketballStats.vegasTotal
    },
    edge: loading ? "-.-%%" : `${basketballStats.edge >= 0 ? '+' : ''}${basketballStats.edge.toFixed(1)}%`,
    pick: {
      record: loading ? "-/-" : `${pickOfDay.record.correct}-${pickOfDay.record.total}`,
      winRate: loading ? "-.-%%" : `${pickOfDay.record.accuracy.toFixed(1)}%`,
      avgOdds: loading ? "-" : formatOdds(pickOfDay.record.avg_odds),
      roi: loading ? "-.-%%" : `${pickOfDay.record.roi >= 0 ? '+' : ''}${pickOfDay.record.roi.toFixed(1)}%`,
      todayPick: pickOfDay.todayPick ? {
        title: pickOfDay.todayPick.matchup,
        prediction: `${pickOfDay.todayPick.picked_team}`,
        odds: formatOdds(pickOfDay.todayPick.picked_odds),
        result: null
      } : null,
      lastPick: pickOfDay.yesterdayPick ? {
        title: pickOfDay.yesterdayPick.matchup,
        prediction: `${pickOfDay.yesterdayPick.picked_team}`,
        odds: formatOdds(pickOfDay.yesterdayPick.picked_odds),
        result: pickOfDay.yesterdayPick.result === 'W' ? "correct" as const :
                pickOfDay.yesterdayPick.result === 'L' ? "incorrect" as const :
                null
      } : null,
      pickLabel: "Yesterday's Pick"
    },
    pickTitle: "Pick of the Day"
  };

  // UFC data
  const ufcData = {
    name: "MMA / UFC",
    subtitle: "Weekly events",
    logo: "/logo/ufc-logo.png",
    path: "/ufc",
    available: true,
    nextEvent: {
      name: "UFC 312: Makhachev vs. Moicano",
      date: "Feb 8, 2025",
      daysAway: 26
    },
    modelAccuracy: {
      winRate: "64.2%",
      record: "82-46",
      totalPicks: 128
    },
    vegasAccuracy: {
      winRate: "58.3%",
      record: "75-53",
      totalPicks: 128
    },
    edge: "+5.9%",
    pick: {
      record: "18-7",
      winRate: "72.0%",
      lastPick: {
        title: "Makhachev vs. Tsarukyan",
        subtitle: "UFC 311",
        prediction: "Makhachev",
        result: "correct" as const
      },
      pickLabel: "Last Week's Pick"
    },
    pickTitle: "Pick of the Week"
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Navbar */}
      <div>
        <Navbar />
        <SportNavBar sports={sports} />
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Sports Sections */}
        <div className="space-y-12">
          <SportSection {...basketballData} />
          <SportSection {...ufcData} />
        </div>
      </div>
    </div>
  );
}
