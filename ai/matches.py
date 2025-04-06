import requests
import csv
import time

# Your API key
API_KEY = "43746d6bdd3b43de8fc5f083e002bb5f"
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"

# Free-tier leagues (Champions League removed)
LEAGUES = {
    "PL": "Premier League",  # England
    "BL1": "Bundesliga",  # Germany
    "SA": "Serie A",  # Italy
    "PD": "La Liga",  # Spain
    "FL1": "Ligue 1"  # France
}

# Seasons to fetch
SEASONS = [2023, 2024]


def fetch_matches(competition_code, season):
    url = f"{BASE_URL}/competitions/{competition_code}/matches?season={season}&status=FINISHED"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        matches = response.json().get("matches", [])
        print(f"Fetched {len(matches)} matches for {competition_code} {season}")
        return matches
    else:
        print(f"Error fetching {competition_code} {season}: {response.status_code}")
        return []


def save_to_csv(matches, filename="matches_2023_2024.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "League", "Home Team", "Away Team", "Result"])

        for match in matches:
            date = match["utcDate"]
            home_team = match["homeTeam"]["name"]
            away_team = match["awayTeam"]["name"]
            winner = match["score"]["winner"]
            league = match["competition"]["name"]
            result = {
                "HOME_TEAM": f"{home_team} Win",
                "AWAY_TEAM": f"{away_team} Win",
                "DRAW": "Draw"
            }.get(winner, "Unknown")

            writer.writerow([date, league, home_team, away_team, result])


def main():
    all_matches = []
    for season in SEASONS:
        for code, name in LEAGUES.items():
            print(f"Fetching {name} for {season}...")
            matches = fetch_matches(code, season)
            all_matches.extend(matches)
            time.sleep(6)  # Avoid rate limit (10 req/min = 1 req/6s)

    # Optional: Limit to 1500 matches
    # all_matches = all_matches[:1500]

    save_to_csv(all_matches)
    print(f"Saved {len(all_matches)} matches to matches_2023_2024.csv")


if __name__ == "__main__":
    main()