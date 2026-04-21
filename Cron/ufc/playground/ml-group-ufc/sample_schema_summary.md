# UFC Fight Data Schema Summary

## Overview
Successfully extracted sample fight data from `fight_snapshots_all_with_rounds.parquet` to visualize the nested DataFrame structure.

## Files Created
1. **sample_fight_clean.json** (16.1 KB) - Clean JSON with properly formatted nested structures
2. **sample_fight_schema.json** (51 KB) - Full sample with all prior fights
3. **extract_sample_fight.py** - Script to extract fights
4. **export_clean_sample.py** - Script to export clean JSON
5. **view_fight_schema.py** - Script to visualize schema

## Sample Fight Details
- **Fight**: BJ Penn vs Caol Uno
- **Date**: 2003-02-28
- **Weight Class**: Lightweight
- **Format**: 5 rounds (title fight)
- **Method**: d_split (Split Decision)
- **Referee**: John McCarthy

## Data Structure

### Top Level (50 fields)
- Fight metadata: fight_id, event_id, fighter IDs, winner/loser
- Fight details: date, link, method, format, type, referee, time, weight class
- Fighter 1 stats: height, weight, reach, stance, DOB, record, striking stats, etc.
- Fighter 2 stats: (same as Fighter 1)
- **prior_f1**: Array of prior fights for Fighter 1 (3 samples included)
- **prior_f2**: Array of prior fights for Fighter 2 (2 samples included)

### Prior Fight Structure (each fight in prior_f1/prior_f2)
Each prior fight contains:
- Fight metadata: fight_id, fight_date, opponent_id, result, method, etc.
- Fight-level stats: body/clinch/ground/head/leg strikes (attempts/landed)
- Control time, knockdowns, significant strikes, takedowns, submissions
- Opponent stats: (same structure with opp_ prefix)
- **rounds**: Array of round-by-round statistics

### Round Structure (each round in rounds array)
Each round contains detailed stats:
- round_number
- kd (knockdowns)
- sig_str_landed, sig_str_attempts
- total_str_landed, total_str_attempts
- td_landed, td_attempts (takedowns)
- sub_att (submission attempts)
- rev (reversals)
- ctrl_time_s (control time in seconds)
- head_landed, head_attempts
- body_landed, body_attempts
- leg_landed, leg_attempts
- distance_landed, distance_attempts
- clinch_landed, clinch_attempts
- ground_landed, ground_attempts

## Notes on Round Data
- Each fight appears to store round data for BOTH fighters in the rounds array
- A 5-round fight shows 10 entries in the rounds array (5 per fighter)
- Round data includes comprehensive striking statistics broken down by:
  - Target (head/body/leg)
  - Position (distance/clinch/ground)
  - Type (significant strikes vs total strikes)

## Round Distribution in Dataset
- 2 rounds: 21 fights
- 4 rounds: 70 fights
- 6 rounds: 3,494 fights (~3 round fights)
- 8 rounds: 94 fights
- 10 rounds: 1,506 fights (~5 round fights)

Total: 5,185 fights in dataset
