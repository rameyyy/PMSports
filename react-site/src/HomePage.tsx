import Navbar from './components/home/Navbar';
import SportNavBar from './components/home/SportNavBar';
import SportSection from './components/home/SportSection';

export default function HomePage() {
  // Sport navigation data
  const sports = [
    { name: "NCAA Men's Basketball", logo: "/logo/ncaa-logo.png", path: "/cbb", available: false },
    { name: "MMA", logo: "/logo/ufc-logo.png", path: "/ufc", available: true }
  ];

  // Basketball data
  const basketballData = {
    name: "NCAA Basketball",
    subtitle: "Daily games (Nov - Apr)",
    logo: "/logo/ncaa-logo.png",
    path: "/cbb",
    available: false,
    nextEvent: {
      name: "Today's Games",
      date: "Jan 13, 2025",
      gamesCount: 12
    },
    modelAccuracy: {
      winRate: "56.4%",
      record: "245-189",
      totalPicks: 434
    },
    vegasAccuracy: {
      winRate: "54.1%",
      record: "235-199",
      totalPicks: 434
    },
    edge: "+2.3%",
    pick: {
      record: "42-28",
      winRate: "60.0%",
      lastPick: {
        title: "Duke vs. North Carolina",
        prediction: "Duke -3.5",
        result: "correct" as const
      },
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
