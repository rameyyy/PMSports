import { useState, useEffect } from 'react';
import Navbar from './components/home/Navbar';
import SportNavBar from './components/home/SportNavBar';
import SportSection from './components/home/SportSection';
import { fetchHomepageStats } from './api/ncaamb';
import type { HomepageStats } from './api/ncaamb';

export default function HomePage() {
  const [stats, setStats] = useState<HomepageStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHomepageStats()
      .then(data => {
        setStats(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch data:', err);
        setLoading(false);
      });
  }, []);

  const sports = [
    { name: "Men's Basketball", logo: "/logo/ncaa-logo.png", path: "/ncaamb", available: true },
    { name: "MMA", logo: "/logo/ufc-logo.png", path: "/ufc", available: true }
  ];

  const formatOdds = (odds: number | null) => {
    if (odds === null) return '-';
    return odds > 0 ? `+${odds}` : `${odds}`;
  };

  const edge = stats ? stats.my_accuracy - stats.vegas_accuracy : 0;

  const basketballData = {
    name: "Men's Basketball",
    subtitle: "Daily games (Nov - Apr)",
    logo: "/logo/ncaa-logo.png",
    path: "/ncaamb",
    available: true,
    nextEvent: {                                                                                                                                                                           
      name: stats?.todays_games_count === -1 ? "-" : "Today's Games",
    // nextEvent: stats?.todays_games_count !== -1 ? {
    //   name: "Today's Games",
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      gamesCount: stats?.todays_games_count === -1 ? 0 : (stats?.todays_games_count || 0)                                                                                                  
    },  
    //   gamesCount: stats?.todays_games_count || 0
    // } : null,
    modelAccuracy: {
      winRate: loading ? "-.-%%" : `${stats?.my_accuracy.toFixed(1)}%`,
      record: loading ? "-/-" : `${stats?.my_total_correct}-${stats?.total_complete_matches}`,
      totalPicks: stats?.total_complete_matches || 0
    },
    vegasAccuracy: {
      winRate: loading ? "-.-%%" : `${stats?.vegas_accuracy.toFixed(1)}%`,
      record: loading ? "-/-" : `${stats?.vegas_total_correct}-${stats?.total_complete_matches}`,
      totalPicks: stats?.total_complete_matches || 0
    },
    edge: loading ? "-.-%%" : `${edge >= 0 ? '+' : ''}${edge.toFixed(1)}%`,
    pick: {
      record: loading ? "-/-" : `${stats?.pick_of_day_correct}-${stats?.pick_of_day_total}`,
      winRate: loading ? "-.-%%" : `${stats?.pick_of_day_acc.toFixed(1)}%`,
      avgOdds: loading ? "-" : formatOdds(stats?.pod_avg_odds || 0),
      roi: loading ? "-.-%%" : `${(stats?.pod_roi || 0) >= 0 ? '+' : ''}${stats?.pod_roi.toFixed(1)}%`,
      todayPick: (stats?.pod_td_matchup && stats?.pod_td_pick && stats?.todays_games_count !== -1) ? {
        title: stats.pod_td_matchup,
        prediction: stats.pod_td_pick,
        odds: formatOdds(stats.pod_td_odds),
        result: null
      } : null,
      lastPick: stats?.pod_yd_matchup && stats?.pod_yd_pick ? {
        title: stats.pod_yd_matchup,
        prediction: stats.pod_yd_pick,
        odds: formatOdds(stats.pod_yd_odds),
        result: stats.pod_yd_outcome === 'W' ? "correct" as const :
                stats.pod_yd_outcome === 'L' ? "incorrect" as const :
                null
      } : null,
      pickLabel: "Yesterday's Pick"
    },
    pickTitle: "Pick of the Day"
  };

  const ufcData = {
    name: "MMA",
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
      <div>
        <Navbar />
        <SportNavBar sports={sports} />
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="space-y-12">
          <SportSection {...basketballData} />
          <SportSection {...ufcData} />
        </div>
      </div>
    </div>
  );
}
