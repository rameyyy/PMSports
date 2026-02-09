interface DateSelectorProps {
  selectedDate: Date;
  onDateChange: (date: Date) => void;
  gameCount?: number;
  totalGameCount?: number;
}

export default function DateSelector({ selectedDate, onDateChange, gameCount, totalGameCount }: DateSelectorProps) {

  const formatDisplayDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const formatInputDate = (date: Date) => {
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
  };

  const goToPreviousDay = () => {
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() - 1);
    onDateChange(newDate);
  };

  const goToNextDay = () => {
    if (isToday) return;
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() + 1);
    onDateChange(newDate);
  };

  const goToToday = () => {
    onDateChange(new Date());
  };

  const handleDateInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newDate = new Date(e.target.value + 'T00:00:00');
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    if (newDate > today) return;
    onDateChange(newDate);
  };

  const isToday = selectedDate.toDateString() === new Date().toDateString();

  return (
    <div className="flex items-center justify-between bg-slate-800/50 rounded-lg px-2 sm:px-4 py-3 sm:py-4 mb-3 sm:mb-4">
      <button
        onClick={goToPreviousDay}
        className="p-1.5 sm:p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
      >
        <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      <div className="flex flex-col items-center">
        {isToday ? (
          <span className="text-[10px] px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded font-medium mb-1">
            Today
          </span>
        ) : (
          <button
            onClick={goToToday}
            className="text-[10px] px-1.5 py-0.5 bg-orange-500/20 text-orange-400 rounded hover:bg-orange-500/30 transition-colors mb-1"
          >
            Go to Today
          </button>
        )}

        {/* Date picker wrapper - input overlays the button for mobile compatibility */}
        <div className="relative">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 hover:border-orange-500 rounded-lg transition-all group">
            <svg className="w-4 h-4 text-slate-400 group-hover:text-orange-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span className="text-white font-semibold text-sm sm:text-lg group-hover:text-orange-400 transition-colors">
              {formatDisplayDate(selectedDate)}
            </span>
          </div>
          <input
            type="date"
            value={formatInputDate(selectedDate)}
            max={formatInputDate(new Date())}
            onChange={handleDateInputChange}
            onClick={(e) => (e.target as HTMLInputElement).showPicker?.()}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
        </div>

        {gameCount !== undefined && (
          <span className="text-xs text-slate-200 mt-1">
            {gameCount === 0 ? null : totalGameCount !== undefined && gameCount !== totalGameCount
              ? `${gameCount}/${totalGameCount} Games`
              : `${gameCount} Game${gameCount !== 1 ? 's' : ''}`}
          </span>
        )}
      </div>

      <button
        onClick={goToNextDay}
        disabled={isToday}
        className={`p-1.5 sm:p-2 rounded-lg transition-colors ${isToday ? 'text-slate-600 cursor-not-allowed' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
      >
        <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>
    </div>
  );
}
