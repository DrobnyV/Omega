import pandas as pd
import re
from datetime import datetime

matches_df = pd.read_csv('../matches_2023_2024.csv')
team_stats_df = pd.read_csv('../all_leagues_2022_2025.csv')


def get_season_from_date(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    year = date.year
    month = date.month
    if month >= 8:
        return f"{year}-{year + 1}"
    else:
        return f"{year - 1}-{year}"

matches_df['Season'] = matches_df['Date'].apply(get_season_from_date)

def find_team_match(match_team, stats_teams):
    if pd.isna(match_team):
        return None
    match_team_lower = match_team.lower()
    for stats_team in stats_teams:
        stats_team_lower = stats_team.lower()
        if (re.search(re.escape(stats_team_lower), match_team_lower) or
                re.search(re.escape(match_team_lower), stats_team_lower)):
            return stats_team
    return None


for league in matches_df['League'].unique():
    for season in matches_df['Season'].unique():
        match_teams = matches_df[(matches_df['League'] == league) &
                                 (matches_df['Season'] == season)][['Home Team', 'Away Team']].stack().unique()
        stats_teams = team_stats_df[(team_stats_df['League'] == league) &
                                    (team_stats_df['Season'] == season)]['Squad'].unique()

        temp_mapping = {}
        unmapped = []
        for match_team in match_teams:
            matched_team = find_team_match(match_team, stats_teams)
            if matched_team:
                temp_mapping[match_team] = matched_team
            else:
                unmapped.append(match_team)

        mask = (matches_df['League'] == league) & (matches_df['Season'] == season)
        matches_df.loc[mask, 'Home Team'] = matches_df.loc[mask, 'Home Team'].map(temp_mapping)
        matches_df.loc[mask, 'Away Team'] = matches_df.loc[mask, 'Away Team'].map(temp_mapping)

        if unmapped:
            print(f"Unmapped teams in {league} {season}: {unmapped}")

matches_df = matches_df.dropna(subset=['Home Team', 'Away Team'])

merged_home = matches_df.merge(
    team_stats_df,
    how='left',
    left_on=['Home Team', 'League', 'Season'],
    right_on=['Squad', 'League', 'Season'],
    suffixes=('', '_home')
).drop(columns=['Squad'])

merged_final = merged_home.merge(
    team_stats_df,
    how='left',
    left_on=['Away Team', 'League', 'Season'],
    right_on=['Squad', 'League', 'Season'],
    suffixes=('_home', '_away')
).drop(columns=['Squad'])

merged_final = merged_final.rename(columns={
    'W_home': 'Home_Wins',
    'D_home': 'Home_Draws',
    'L_home': 'Home_Losses',
    'Pts/MP_home': 'Home_PtsPerMatch',
    'GD_home': 'Home_GoalDiff',
    'xGD_home': 'Home_xGD',
    'W_away': 'Away_Wins',
    'D_away': 'Away_Draws',
    'L_away': 'Away_Losses',
    'Pts/MP_away': 'Away_PtsPerMatch',
    'GD_away': 'Away_GoalDiff',
    'xGD_away': 'Away_xGD'
})

final_columns = [
    'Date', 'League', 'Season', 'Home Team', 'Away Team', 'Result',
    'Home_Wins', 'Home_Draws', 'Home_Losses', 'Home_PtsPerMatch', 'Home_GoalDiff', 'Home_xGD',
    'Away_Wins', 'Away_Draws', 'Away_Losses', 'Away_PtsPerMatch', 'Away_GoalDiff', 'Away_xGD'
]
training_df = merged_final[final_columns]


def standardize_result(row):
    result = row['Result']
    home_team = row['Home Team']
    away_team = row['Away Team']

    if result == "Draw":
        return "Draw"
    if "Borussia Mönchengladbach" in result and home_team == "Gladbach":
        return "Home Win"
    if "Borussia Mönchengladbach" in result and away_team == "Gladbach":
        return "Away Win"
    if home_team in result:
        return "Home Win"
    if away_team in result:
        return "Away Win"
    return result


training_df['Result'] = training_df.apply(standardize_result, axis=1)

unmapped_results = training_df[~training_df['Result'].isin(['Draw', 'Home Win', 'Away Win'])]['Result']
if not unmapped_results.empty:
    print("Unmapped Result values found:")
    print(unmapped_results.unique())

training_df.to_csv('training_data_2023_2024.csv', index=False)
print("Unique Result values after standardization:")
print(training_df['Result'].unique())


print("Merged Dataset Preview:")
print(training_df.head())
print(f"\nSaved {len(training_df)} rows to 'training_data_2023_2024.csv'")