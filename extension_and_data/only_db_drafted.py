import pandas as pd
import re
import unicodedata


INPUT_CSV = "combine_with_college_stats.csv"
OUTPUT_CSV = "defensive_backs_drafted.csv"

# Positions to keep as defensive backs
DB_POSITIONS = {"CB", "DB", "FS", "SS", "S", "NB", "SAF"}

# Optional mappings if your source uses odd labels
POSITION_NORMALIZATION = {
    "FREE SAFETY": "FS",
    "STRONG SAFETY": "SS",
    "SAFETY": "S",
    "CORNERBACK": "CB",
    "DEFENSIVE BACK": "DB",
    "NICKEL BACK": "NB",
}


def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def normalize_position(pos: str) -> str:
    pos = normalize_text(pos).upper()
    pos = POSITION_NORMALIZATION.get(pos, pos)
    return pos


def is_defensive_back(pos: str) -> bool:
    return normalize_position(pos) in DB_POSITIONS


def extract_draft_year(drafted_field: str):
    """
    Example:
    'New York Jets / 1st / 13th pick / 2000' -> 2000
    """
    drafted_field = normalize_text(drafted_field)
    match = re.search(r"\b(19|20)\d{2}\b", drafted_field)
    return int(match.group(0)) if match else None


def was_drafted(drafted_field: str) -> bool:
    drafted_field = normalize_text(drafted_field)
    if not drafted_field:
        return False
    return extract_draft_year(drafted_field) is not None


def slugify_name(name: str) -> str:
    """
    Sports Reference style slug:
    "A'Shawn Robinson" -> "ashawn-robinson"
    "A.J. Terrell" -> "aj-terrell"
    """
    name = normalize_text(name).lower()
    name = name.replace("'", "")
    name = name.replace(".", "")
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


def make_sportsref_predicted_url(player_name: str) -> str:
    slug = slugify_name(player_name)
    return f"https://www.sports-reference.com/cfb/players/{slug}-1.html"


def main():
    df = pd.read_csv(INPUT_CSV)

    # Strip column names just in case
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["Player", "Pos", "Drafted (tm/rnd/yr)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean important columns
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Pos"] = df["Pos"].astype(str).str.strip()
    df["Drafted (tm/rnd/yr)"] = df["Drafted (tm/rnd/yr)"].fillna("").astype(str).str.strip()

    # Keep drafted players only
    df = df[df["Drafted (tm/rnd/yr)"].apply(was_drafted)].copy()

    # Keep defensive backs only
    df["Pos"] = df["Pos"].apply(normalize_position)
    df = df[df["Pos"].apply(is_defensive_back)].copy()

    # Extract draft year
    df["year"] = df["Drafted (tm/rnd/yr)"].apply(extract_draft_year)

    # Remove duplicates
    # NFL_id is best if present
    if "NFL_id" in df.columns:
        df = df.drop_duplicates(subset=["NFL_id"])
    else:
        df = df.drop_duplicates(subset=["Player", "year", "Pos"])

    # Generate predicted Sports Reference URLs
    df["sportsref_predicted_url"] = df["Player"].apply(make_sportsref_predicted_url)

    # Output format
    output_df = df[["Player", "Pos", "year", "sportsref_predicted_url"]].copy()

    # Sort for readability
    output_df = output_df.sort_values(by=["year", "Player"]).reset_index(drop=True)

    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(output_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()