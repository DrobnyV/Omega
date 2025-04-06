import pandas as pd
import time

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

leagues = {
    "Premier League": 9,
    "Bundesliga": 20,
    "Serie A": 11,
    "La Liga": 12,
    "Ligue 1": 13
}

seasons = ["2022-2023", "2023-2024", "2024-2025"]

columns = ['Squad', 'W', 'D', 'L', 'Pts/MP', 'GD', 'xGD']

all_team_stats = []

for league_name, comp_id in leagues.items():
    for season in seasons:
        url = f"https://fbref.com/en/comps/{comp_id}/{season}/{season}-{league_name.replace(' ', '-')}-Stats"
        table_id = f"results{season}{comp_id}1_overall"

        print(f"Fetching {league_name} for {season} from {url} (Table ID: {table_id})...")
        try:
            df = pd.read_html(url, attrs={"id": table_id})[0]
            team_stats = df[columns].copy()
            team_stats['Season'] = season
            team_stats['League'] = league_name
            all_team_stats.append(team_stats)
            print(f"Fetched {len(team_stats)} teams for {league_name} {season}")
        except Exception as e:
            print(f"Error fetching {league_name} {season}: {e}")
        time.sleep(3)


combined_team_stats = pd.concat(all_team_stats, ignore_index=True)


combined_team_stats.to_csv('all_leagues_2022_2025.csv', index=False)


print("\nCombined Stats for All Leagues (2022-2023 to 2024-2025):")
print(combined_team_stats)

print(f"\nSaved data to 'all_leagues_2022_2025.csv' with {len(combined_team_stats)} rows.")