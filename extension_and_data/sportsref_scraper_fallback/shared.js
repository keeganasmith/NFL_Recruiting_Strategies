const SPORTSREF_ORIGIN = "https://www.sports-reference.com";

function normalizeTextAscii(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "");
}

function slugifyName(name) {
  return normalizeTextAscii(name)
    .toLowerCase()
    .replace(/'/g, "")
    .replace(/\./g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function isLikelyPfrPlayerId(value) {
  const raw = String(value || "").trim();
  return /^[A-Za-z]{4}[A-Za-z]{2}\d{2}$/.test(raw);
}

function selectSportsRefSlugSource(row, playerName) {
  const candidates = [
    row && row.sportsref_url,
    row && row.sportsref_predicted_url,
    row && row.slug,
    playerName
  ];

  for (const candidate of candidates) {
    const value = String(candidate || "").trim();
    if (!value) continue;
    if (isLikelyPfrPlayerId(value)) continue;
    return value;
  }

  return "";
}

function normalizeSlugBase(slugOrName) {
  const raw = String(slugOrName || "").trim().toLowerCase();
  if (!raw) return "";

  const withoutHtml = raw.replace(/\.html$/i, "").replace(/\/+$/g, "");
  const fromUrl = withoutHtml
    .replace(/^https?:\/\/[^/]+/i, "")
    .replace(/^.*\/cfb\/players\//, "");
  const withoutSuffix = fromUrl.replace(/-\d+$/i, "");
  return slugifyName(withoutSuffix);
}

function computePlayerUrl(slugOrName, attemptIndex) {
  const safeAttempt = Number.isInteger(attemptIndex) && attemptIndex > 0 ? attemptIndex : 1;
  const base = normalizeSlugBase(slugOrName);
  if (!base) return "";
  return `${SPORTSREF_ORIGIN}/cfb/players/${base}-${safeAttempt}.html`;
}
