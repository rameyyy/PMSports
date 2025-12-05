CREATE TABLE moneyline (
  id INT AUTO_INCREMENT PRIMARY KEY,
  game_id VARCHAR(20) NOT NULL,
  team_1 VARCHAR(100) NOT NULL,
  team_2 VARCHAR(100) NOT NULL,
  team_predicted_to_win VARCHAR(100) NOT NULL,

  xgb_prob_team_1 DECIMAL(5, 4) NOT NULL,
  xgb_prob_team_2 DECIMAL(5, 4) NOT NULL,
  gbm_prob_team_1 DECIMAL(5, 4) NOT NULL,
  gbm_prob_team_2 DECIMAL(5, 4) NOT NULL,
  ensemble_prob_team_1 DECIMAL(5, 4) NOT NULL,
  ensemble_prob_team_2 DECIMAL(5, 4) NOT NULL,

  best_ev_team_1 DECIMAL(8, 4),
  best_ev_team_2 DECIMAL(8, 4),
  my_best_ev_team_1 DECIMAL(8, 4),
  my_best_ev_team_2 DECIMAL(8, 4),

  best_book_team_1 VARCHAR(50),
  best_book_odds_team_1 INT,
  best_book_team_2 VARCHAR(50),
  best_book_odds_team_2 INT,

  my_best_book_team_1 VARCHAR(50),
  my_best_book_odds_team_1 INT,
  my_best_book_team_2 VARCHAR(50),
  my_best_book_odds_team_2 INT,

  implied_prob_team_1_with_vig DECIMAL(5, 4),
  implied_prob_team_2_with_vig DECIMAL(5, 4),
  implied_prob_team_1_devigged DECIMAL(5, 4),
  implied_prob_team_2_devigged DECIMAL(5, 4),

  bet_rule VARCHAR(255),
  bet_on TINYINT(1) DEFAULT 0,
  wager DECIMAL(10, 2),

  winning_team VARCHAR(100),
  actual_score_team_1 INT,
  actual_score_team_2 INT,
  actual_total INT,

  bet_win_or_lose VARCHAR(20),
  profit DECIMAL(10, 2),

  game_date DATE NOT NULL,
  season INT NOT NULL,

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

  UNIQUE KEY unique_game (game_id, team_predicted_to_win),
  INDEX idx_game_date (game_date),
  INDEX idx_season (season),
  INDEX idx_bet_on (bet_on),
  INDEX idx_game_id (game_id)
);
