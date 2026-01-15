interface Tab {
  id: string;
  label: string;
}

interface TabNavProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export default function TabNav({ tabs, activeTab, onTabChange }: TabNavProps) {
  return (
    <div className="border-b border-slate-700 bg-slate-800 sticky top-16 z-40">
      <div className="px-4 sm:px-8 flex items-center space-x-1 sm:space-x-4 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`py-4 px-2 sm:px-3 text-xs sm:text-sm font-semibold transition-colors border-b-2 whitespace-nowrap ${
              activeTab === tab.id
                ? 'text-orange-500 border-orange-500 bg-slate-700/20'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
}
