import pandas as pd

EXPECTED_FILE = "defensive_backs_drafted.csv"
SCRAPED_FILE = "sportsref_final_season_data (7).csv"

# Load files
expected_df = pd.read_csv(EXPECTED_FILE)
scraped_df = pd.read_csv(SCRAPED_FILE)

# Clean names
def clean_name(name):
    return str(name).strip().lower()

expected_df["player_clean"] = expected_df["Player"].apply(clean_name)
scraped_df["player_clean"] = scraped_df["player"].apply(clean_name)
expected_df = expected_df.rename(columns={
    "Player": "player",
    "Pos": "pos_expected"
    })

scraped_df = scraped_df.rename(columns={
    "pos": "pos_scraped"   # ← your scraped file uses "class"
})

# -------------------------
# 1. Missing players
# -------------------------
expected_players = set(expected_df["player_clean"])
scraped_players = set(scraped_df["player_clean"])

missing_players = expected_players - scraped_players

missing_df = expected_df[expected_df["player_clean"].isin(missing_players)]

# -------------------------
# 2. Wrong matches (same name but bad position or year)
# -------------------------
# Merge on name
merged = expected_df.merge(
    scraped_df,
    on="player_clean",
    how="inner",
    suffixes=("_expected", "_scraped")
)

DB_GROUP = {"CB", "DB", "FS", "SS", "S", "NB"}

def is_pos_match(expected, scraped):
    expected = str(expected).upper()
    scraped = str(scraped).upper()

    # normalize scraped (split if needed like "CB/DB")
    scraped_parts = set(scraped.replace(",", "/").split("/"))

    # If both are DB-type → valid
    if expected in DB_GROUP and any(p in DB_GROUP for p in scraped_parts):
        return True

    return expected in scraped_parts


def is_mismatch(row):
    # Position mismatch  

    # pos_ok = pos_expected in pos_scraped
    pos_ok = is_pos_match(row["pos_expected"], row["pos_scraped"])
    # Year check: final_season + 1 == draft year
    try:
        year_ok = int(row["year"]) == int(row["final_season_year"]) + 1 or int(row["year"] == 2025)
    except:
        year_ok = False

    return not (pos_ok and year_ok)

wrong_matches = merged[merged.apply(is_mismatch, axis=1)]

# -------------------------
# 3. Stats
# -------------------------
total_expected = len(expected_df)
total_scraped = len(scraped_players)
total_missing = len(missing_df)
total_wrong = len(wrong_matches)

# -------------------------
# 4. Save outputs
# -------------------------
missing_df[["player", "pos_expected", "year", "sportsref_predicted_url"]].to_csv(
    "missing_players.csv", index=False
)

wrong_matches.to_csv("wrong_matches.csv", index=False)

# -------------------------
# 5. Print summary
# -------------------------
print("===== RESULTS =====")
print(f"Total expected: {total_expected}")
print(f"Total scraped: {total_scraped}")
print(f"Missing players: {total_missing}")
print(f"Wrong matches: {total_wrong}")

print("\nSaved:")
print("- missing_players.csv")
print("- wrong_matches.csv")