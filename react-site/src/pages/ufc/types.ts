export type Fight = {
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
  f1_probability: number | null;
  predicted_winner_id: string | null;
  correct: number | null;
  date: string;
  weight_class: string;
  fight_type: string | null;
  f1_odds: number | null;
  f2_odds: number | null;
  win_method: string | null;
  end_time: string | null;
  actual_winner_id: string | null;
};

export type Event = {
  event_id: string;
  event_url: string;
  title: string;
  event_datestr: string;
  location: string;
  date: string;
};
