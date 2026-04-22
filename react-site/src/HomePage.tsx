import { useState, useEffect } from 'react';
import Navbar from './components/home/Navbar';
import SportNavBar from './components/home/SportNavBar';
import SportSection from './components/home/SportSection';
import { fetchHomepageStats } from './api/ncaamb';
import { fetchUFCHomepageStats } from './api/ufc';
import type { HomepageStats } from './api/ncaamb';
import type { UFCHomepageStats } from './api/ufc';

export default function HomePage() {
  const [stats, setStats] = useState<HomepageStats | null>(null);
  const [ufcStats, setUfcStats] = useState<UFCHomepageStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [ufcLoading, setUfcLoading] = useState(true);

  useEffect(() => {
    fetchHomepageStats()
      .then(data => { setStats(data); setLoading(false); })
      .catch(err => { console.error('Failed to fetch NCAAMB data:', err); setLoading(false); });

    fetchUFCHomepageStats()
      .then(data => { setUfcStats(data); setUfcLoading(false); })
      .catch(err => { console.error('Failed to fetch UFC data:', err); setUfcLoading(false); });
  }, []);

  const sports = [
    { name: "Men's Basketball", logo: "/logo/ncaa-logo.png", path: "/ncaamb", available: true },
    { name: "MMA", logo: "/logo/ufc-logo.png", path: "/ufc", available: true }
  ];

  const formatOdds = (odds: number | null) => {
    if (odds === null) return '-';
    return odds > 0 ? `+${odds}` : `${odds}`;
  };

  const capitalizeName = (name: string) =>
    name.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');

  const capitalizeMatchup = (matchup: string | null | undefined) => {
    if (!matchup) return matchup;
    return matchup.split(/ vs\.? /i).map(capitalizeName).join(' vs. ');
  };

  // Calculate edge from rounded values (to match what's displayed)
  const myModelRounded = stats ? Math.round(stats.my_accuracy * 10) / 10 : 0;
  const vegasRounded = stats ? Math.round(stats.vegas_accuracy * 10) / 10 : 0;
  const edge = myModelRounded - vegasRounded;

  const isSeasonOver = stats?.todays_games_count === -1;

  const basketballData = {
    name: "Men's Basketball",
    subtitle: isSeasonOver ? "2026 Season Complete" : "Daily games (Nov - Apr)",
    logo: "/logo/ncaa-logo.png",
    path: "/ncaamb",
    available: true,
    nextEvent: isSeasonOver ? {
      name: "Season Over",
      date: "Returns November 2026",
      gamesCount: 0
    } : {
      name: "Today's Games",
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      gamesCount: stats?.todays_games_count || 0
    },
    modelAccuracy: {
      winRate: loading ? "-.-%%" : `${stats?.my_accuracy.toFixed(1)}%`,
      record: loading ? "-/-" : `${stats?.my_total_correct}-${stats?.total_complete_matches}`,
      totalPicks: stats?.total_complete_matches || 0,
      label: isSeasonOver ? "2026 Season" : undefined
    },
    vegasAccuracy: {
      winRate: loading ? "-.-%%" : `${stats?.vegas_accuracy.toFixed(1)}%`,
      record: loading ? "-/-" : `${stats?.vegas_total_correct}-${stats?.total_complete_matches}`,
      totalPicks: stats?.total_complete_matches || 0,
      label: isSeasonOver ? "2026 Season" : undefined
    },
    edge: loading ? "-.-%%" : `${edge >= 0 ? '+' : ''}${edge.toFixed(1)}%`,
    pick: {
      record: loading ? "-/-" : `${stats?.pick_of_day_correct}-${stats?.pick_of_day_total}`,
      winRate: loading ? "-.-%%" : `${stats?.pick_of_day_acc.toFixed(1)}%`,
      avgOdds: loading ? "-" : formatOdds(stats?.pod_avg_odds || 0),
      units: loading ? "-" : `${(stats?.pod_units || 0) >= 0 ? '+' : ''}${stats?.pod_units.toFixed(2)}u`,
      todayPick: (stats?.pod_td_matchup && stats?.pod_td_pick && !isSeasonOver) ? {
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
      pickLabel: "Yesterday's Pick",
      seasonLabel: isSeasonOver ? "2026 Season Stats" : undefined
    },
    pickTitle: "Pick of the Day"
  };

  const ufcModelAcc  = ufcStats?.model_accuracy;
  const ufcVegasAcc  = ufcStats?.vegas_accuracy;
  const ufcPOW       = ufcStats?.pick_of_week;
  const ufcLastPick  = ufcStats?.last_pick;
  const ufcNextPick  = ufcStats?.next_pick;
  const ufcNextEvent = ufcStats?.next_event;

  const ufcModelCorrect = ufcModelAcc ? ufcModelAcc.total - ufcModelAcc.correct : 0;
  const ufcVegasCorrect = ufcVegasAcc ? ufcVegasAcc.total - ufcVegasAcc.correct : 0;
  const ufcEdge = (ufcModelAcc && ufcVegasAcc)
    ? ufcModelAcc.accuracy - ufcVegasAcc.accuracy
    : null;

  const ufcData = {
    name: "MMA",
    subtitle: "Weekly events",
    logo: "/logo/ufc-logo.png",
    path: "/ufc",
    available: true,
    nextEvent: {
      name: ufcNextEvent?.name ?? "—",
      date: ufcNextEvent?.date
        ? new Date(ufcNextEvent.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
        : "—",
      daysAway: ufcNextEvent?.days_away ?? undefined,
    },
    modelAccuracy: {
      winRate: ufcLoading ? "-.-%" : `${ufcModelAcc?.accuracy.toFixed(1)}%`,
      record:  ufcLoading ? "-/-"  : `${ufcModelAcc?.correct}-${ufcModelCorrect}`,
      totalPicks: ufcModelAcc?.total ?? 0,
    },
    vegasAccuracy: {
      winRate: ufcLoading ? "-.-%" : `${ufcVegasAcc?.accuracy.toFixed(1)}%`,
      record:  ufcLoading ? "-/-"  : `${ufcVegasAcc?.correct}-${ufcVegasCorrect}`,
      totalPicks: ufcVegasAcc?.total ?? 0,
    },
    edge: ufcLoading || ufcEdge === null
      ? "-.-%"
      : `${ufcEdge >= 0 ? '+' : ''}${ufcEdge.toFixed(1)}%`,
    pick: {
      record:   ufcLoading ? "-/-"  : (ufcPOW?.record ?? "-/-"),
      winRate:  ufcLoading ? "-.-%" : `${ufcPOW?.win_rate.toFixed(1)}%`,
      avgOdds:  (!ufcLoading && ufcPOW?.avg_odds != null) ? formatOdds(ufcPOW.avg_odds) : undefined,
      units:    (!ufcLoading && ufcPOW?.units != null) ? `${ufcPOW.units >= 0 ? '+' : ''}${ufcPOW.units.toFixed(2)}u` : undefined,
      todayPick: (ufcNextPick && !ufcLoading) ? {
        title:      capitalizeMatchup(ufcNextPick.matchup) ?? "",
        subtitle:   ufcNextPick.event ?? undefined,
        prediction: capitalizeName(ufcNextPick.prediction ?? ""),
        odds:       ufcNextPick.odds != null ? formatOdds(ufcNextPick.odds) : undefined,
        result:     null,
      } : null,
      todayPickLabel: "This Week's Pick",
      lastPick: (ufcLastPick && !ufcLoading) ? {
        title:      capitalizeMatchup(ufcLastPick.matchup) ?? "",
        subtitle:   ufcLastPick.event ?? undefined,
        prediction: capitalizeName(ufcLastPick.prediction ?? ""),
        odds:       ufcLastPick.odds != null ? formatOdds(ufcLastPick.odds) : undefined,
        result:     ufcLastPick.correct === true  ? "correct" as const
                  : ufcLastPick.correct === false ? "incorrect" as const
                  : null,
      } : null,
      pickLabel: "Last Week's Pick",
    },
    pickTitle: "Pick of the Week",
  };

  // Reorder sports based on season status
  const orderedSports = isSeasonOver
    ? [ufcData, basketballData]
    : [basketballData, ufcData];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div>
        <Navbar />
        <SportNavBar sports={sports} />
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="space-y-12">
          {orderedSports.map((sport) => (
            <SportSection key={sport.name} {...sport} />
          ))}
        </div>
      </div>
    </div>
  );
}
