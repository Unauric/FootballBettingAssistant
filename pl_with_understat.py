import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LEAGUE = "EPL"
SEASONS = list(range(2020, 2025))          # match your cache seasons
FOOTBALL_FILES = {s: f"E{s}.csv" for s in SEASONS}

CACHE_DIR = Path("understat_cache")

MIN_TRAIN_MATCHES = 160
RETRAIN_EVERY = 15
HALF_LIFE_DAYS = 60

N_MODELS = 15
MODEL_SEED = 42
P_LOW_Q = 0.10

EXCLUDE_PREFIXES = {"Avg", "AvgC", "Max", "MaxC", "1XB", "1XBC"}
MIN_P_LOW = 0.18
EV_LOW_THRESHOLD = 0.03
STAKE = 1.0

WIN_FORM = 5
WIN_SHOTS = 8

OUTCOMES = ["H", "D", "A"]
LABEL_MAP = {"H": 0, "D": 1, "A": 2}

P_CLIP_LOW = 0.01
P_CLIP_HIGH = 0.98

SOT_RESULTS = {"Goal", "SavedShot"} 
SETPIECE_SITUATIONS = {"FromCorner", "DirectFreekick", "SetPiece"}
PENALTY_SITUATION = "Penalty"

SHOT_KEYS = [
    "shots", "sot_rate",
    "npxg_sum", "xg_per_shot",
    "big", "vbig",
    "box_shots", "setpiece_shots",
    "late_xg",
]

def safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def normalize_team(name: str) -> str:
    s = str(name).strip()
    s = s.replace("Nott'm Forest", "Nottingham Forest")
    s = s.replace("Man United", "Manchester United")
    s = s.replace("Man City", "Manchester City")
    s = s.replace("Spurs", "Tottenham")
    s = s.replace("Wolves", "Wolverhampton Wanderers")
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class BookTriplet:
    prefix: str
    h: str
    d: str
    a: str


def detect_book_triplets(df: pd.DataFrame) -> List[BookTriplet]:
    cols = list(df.columns)
    by_prefix: Dict[str, Dict[str, str]] = {}

    for c in cols:
        m = re.match(r"^([A-Za-z0-9]+)([HDA])$", c)
        if not m:
            continue
        prefix, suf = m.group(1), m.group(2)
        by_prefix.setdefault(prefix, {})[suf] = c

    triplets: List[BookTriplet] = []
    for prefix, dct in by_prefix.items():
        if prefix in EXCLUDE_PREFIXES:
            continue
        if not all(k in dct for k in ("H", "D", "A")):
            continue
        h, d, a = dct["H"], dct["D"], dct["A"]
        sample = pd.to_numeric(df[h], errors="coerce")
        med = float(sample.dropna().median()) if sample.notna().any() else np.nan
        if np.isfinite(med) and 1.01 <= med <= 100.0:
            triplets.append(BookTriplet(prefix=prefix, h=h, d=d, a=a))

    return sorted(triplets, key=lambda t: t.prefix)


def best_odds_and_book(row, triplets: List[BookTriplet], outcome: str) -> Tuple[float, Optional[str]]:
    best = np.nan
    best_book = None
    for t in triplets:
        col = t.h if outcome == "H" else t.d if outcome == "D" else t.a
        v = safe_float(row.get(col, np.nan))
        if np.isfinite(v) and v > 1.01:
            if (not np.isfinite(best)) or (v > best):
                best = v
                best_book = t.prefix
    return best, best_book


def market_consensus_probs(row, triplets: List[BookTriplet]) -> Tuple[float, float, float]:
    phs, pds, pas = [], [], []
    for t in triplets:
        oh = safe_float(row.get(t.h, np.nan))
        od = safe_float(row.get(t.d, np.nan))
        oa = safe_float(row.get(t.a, np.nan))
        if not (np.isfinite(oh) and np.isfinite(od) and np.isfinite(oa)):
            continue
        if oh <= 1.01 or od <= 1.01 or oa <= 1.01:
            continue
        ph, pd_, pa = 1/oh, 1/od, 1/oa
        s = ph + pd_ + pa
        if s <= 0:
            continue
        phs.append(ph/s); pds.append(pd_/s); pas.append(pa/s)

    if not phs:
        return np.nan, np.nan, np.nan

    ph = float(np.mean(phs)); pd_ = float(np.mean(pds)); pa = float(np.mean(pas))
    s = ph + pd_ + pa
    return (ph/s, pd_/s, pa/s) if s > 0 else (np.nan, np.nan, np.nan)


def time_decay_weights(train_dates: np.ndarray, ref_date: np.datetime64, half_life_days: float) -> np.ndarray:
    age_days = (ref_date - train_dates).astype("timedelta64[D]").astype(float)
    age_days = np.maximum(age_days, 0.0)
    lam = np.log(2.0) / float(half_life_days)
    return np.exp(-lam * age_days)


def agg_side(shots_list: List[dict]) -> Dict[str, float]:
    shots = len(shots_list)

    xg_sum = 0.0
    npxg_sum = 0.0
    sots = 0

    big = 0        
    vbig = 0       
    box_shots = 0  
    setpiece_shots = 0
    late_xg = 0.0 

    for s in shots_list:
        xg = safe_float(s.get("xG", 0.0))
        if not np.isfinite(xg): xg = 0.0

        minute = safe_float(s.get("minute", 0.0))
        if not np.isfinite(minute): minute = 0.0

        res = str(s.get("result", ""))
        sit = str(s.get("situation", ""))

        X = safe_float(s.get("X", 0.0))
        if not np.isfinite(X): X = 0.0

        xg_sum += xg

        if res in SOT_RESULTS:
            sots += 1

        if sit != PENALTY_SITUATION:
            npxg_sum += xg

        if xg >= 0.10:
            big += 1
        if xg >= 0.30:
            vbig += 1

        if X >= 0.83:
            box_shots += 1

        if sit in SETPIECE_SITUATIONS:
            setpiece_shots += 1

        if minute >= 75:
            late_xg += xg

    xg_per_shot = xg_sum / shots if shots > 0 else 0.0
    sot_rate = sots / shots if shots > 0 else 0.0

    return {
        "shots": float(shots),
        "sot_rate": float(sot_rate),
        "xg_sum": float(xg_sum),
        "npxg_sum": float(npxg_sum),
        "xg_per_shot": float(xg_per_shot),
        "big": float(big),
        "vbig": float(vbig),
        "box_shots": float(box_shots),
        "setpiece_shots": float(setpiece_shots),
        "late_xg": float(late_xg),
    }


def agg_match_shots(obj: Dict[str, any]) -> Dict[str, float]:
    h_list = obj.get("h") or []
    a_list = obj.get("a") or []
    H = agg_side(h_list)
    A = agg_side(a_list)

    out = {f"US_H_{k}": v for k, v in H.items()}
    out.update({f"US_A_{k}": v for k, v in A.items()})
    return out


def load_understat_season_index(season: int) -> pd.DataFrame:
    p = CACHE_DIR / f"{LEAGUE}_{season}_matches.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} - run your caching script first.")

    matches = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for m in matches:
        if m.get("isResult") is not True:
            continue
        dt = m.get("datetime") or m.get("date")
        d = pd.to_datetime(dt, errors="coerce")
        if pd.isna(d):
            continue
        h = m.get("h") or {}
        a = m.get("a") or {}
        home = normalize_team(h.get("title") or h.get("name"))
        away = normalize_team(a.get("title") or a.get("name"))
        mid = m.get("id")
        if mid is None:
            continue
        rows.append({
            "US_match_id": str(mid),
            "Date": d.normalize(),
            "HomeTeamN": home,
            "AwayTeamN": away,
        })
    return pd.DataFrame(rows)


def attach_shot_features(season_df: pd.DataFrame, us_index: pd.DataFrame) -> pd.DataFrame:
    """
    Merge shot-level aggregates into the football-data matches for a season.
    We do it by matching (Date, HomeTeamN, AwayTeamN) -> US_match_id, then reading match_<id>_shots.json
    """
    dfm = season_df.merge(us_index, how="left", on=["Date", "HomeTeamN", "AwayTeamN"])

    feats = []
    have = 0

    for mid in dfm["US_match_id"].astype(str).tolist():
        if mid == "nan" or mid == "None":
            feats.append({})
            continue

        fp = CACHE_DIR / f"match_{mid}_shots.json"
        if not fp.exists():
            feats.append({})
            continue

        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj.get("_error"):
                feats.append({})
                continue
            if not isinstance(obj, dict):
                feats.append({})
                continue
            feats.append(agg_match_shots(obj))
            have += 1
        except Exception:
            feats.append({})

    feats_df = pd.DataFrame(feats).fillna(0.0)
    out = pd.concat([dfm.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    print(f"[info] Season {int(season_df['Season'].iloc[0])} shot feature coverage: {have}/{len(out)} = {have/len(out):.1%}")
    return out



def add_rolling_team_form(df: pd.DataFrame) -> pd.DataFrame:
    teams = pd.unique(df[["HomeTeamN", "AwayTeamN"]].values.ravel())
    hist = {t: [] for t in teams}

    def mean_last(team: str, key: str, n: int) -> float:
        arr = [r.get(key, 0.0) for r in hist[team][-n:]]
        return float(np.mean(arr)) if arr else 0.0

    rows = []
    for _, row in df.iterrows():
        ht, at = row["HomeTeamN"], row["AwayTeamN"]

        feats = {
            "home_pts": mean_last(ht, "pts", WIN_FORM),
            "away_pts": mean_last(at, "pts", WIN_FORM),
        }

        for k in SHOT_KEYS:
            feats[f"home_{k}_r{WIN_SHOTS}"] = mean_last(ht, k, WIN_SHOTS)
            feats[f"away_{k}_r{WIN_SHOTS}"] = mean_last(at, k, WIN_SHOTS)

        rows.append(feats)

        # update AFTER match
        ftr = row["FTR"]
        if ftr == "H":
            hpts, apts = 3, 0
        elif ftr == "A":
            hpts, apts = 0, 3
        else:
            hpts, apts = 1, 1

        def side(prefix: str, pts: int) -> Dict[str, float]:
            d = {"pts": float(pts)}
            # realized match shot aggregates
            for k in SHOT_KEYS:
                d[k] = safe_float(row.get(prefix + k, 0.0))
                if not np.isfinite(d[k]):
                    d[k] = 0.0
            return d

        hist[ht].append(side("US_H_", hpts))
        hist[at].append(side("US_A_", apts))

    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)



def make_model(seed: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            loss="log_loss",
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1e-3,
            random_state=seed
        ))
    ])


def fit_ensemble(X, y, sw, n_models: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    models = []
    for k in range(n_models):
        idx = rng.integers(0, n, size=n, endpoint=False)
        m = make_model(seed + k + 1)
        m.fit(X[idx], y[idx], clf__sample_weight=sw[idx])
        models.append(m)
    return models



def load_football_season(season: int) -> pd.DataFrame:
    f = FOOTBALL_FILES[season]
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f} in current folder.")

    df = pd.read_csv(f)
    df["Season"] = season
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"]).sort_values("Date").reset_index(drop=True)
    df["HomeTeamN"] = df["HomeTeam"].apply(normalize_team)
    df["AwayTeamN"] = df["AwayTeam"].apply(normalize_team)
    return df


def backtest_season(season: int) -> Tuple[int, float, float, pd.DataFrame]:
    df = load_football_season(season)

    triplets = detect_book_triplets(df)
    if not triplets:
        print(f"[warn] No odds triplets detected for season {season}, skipping.")
        return 0, 0.0, 0.0, pd.DataFrame()

    cons = df.apply(lambda r: market_consensus_probs(r, triplets), axis=1, result_type="expand")
    cons.columns = ["mkt_pH", "mkt_pD", "mkt_pA"]
    df = pd.concat([df, cons], axis=1)

    us_index = load_understat_season_index(season)
    df = attach_shot_features(df, us_index)

    df = df.sort_values("Date").reset_index(drop=True)
    df = add_rolling_team_form(df)

    df = df[df["FTR"].isin(LABEL_MAP)].copy()
    df["y"] = df["FTR"].map(LABEL_MAP).astype(int)

    # feature columns
    feature_cols = ["mkt_pH", "mkt_pD", "mkt_pA", "home_pts", "away_pts"]
    for k in SHOT_KEYS:
        feature_cols.append(f"home_{k}_r{WIN_SHOTS}")
        feature_cols.append(f"away_{k}_r{WIN_SHOTS}")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = df[feature_cols].to_numpy(float)
    y = df["y"].to_numpy(int)
    dates = df["Date"].to_numpy("datetime64[ns]")

    profits = []
    picks = []

    ensemble = None
    last_fit = -1

    for i in range(len(df)):
        if i < MIN_TRAIN_MATCHES:
            continue

        if ensemble is None or (i - last_fit) >= RETRAIN_EVERY:
            ref = dates[i]
            sw = time_decay_weights(dates[:i], ref, HALF_LIFE_DAYS)
            ensemble = fit_ensemble(X[:i], y[:i], sw, N_MODELS, MODEL_SEED)
            last_fit = i

        # ensemble probability samples
        proba_samples = np.stack([m.predict_proba(X[i:i+1])[0] for m in ensemble], axis=0)
        proba_samples = np.clip(proba_samples, P_CLIP_LOW, P_CLIP_HIGH)

        p_low = np.quantile(proba_samples, P_LOW_Q, axis=0)
        p_med = np.quantile(proba_samples, 0.50, axis=0)

        row = df.iloc[i]
        best = None

        for oi, outcome in enumerate(OUTCOMES):
            odds, book = best_odds_and_book(row, triplets, outcome)
            if not np.isfinite(odds):
                continue

            pl = float(p_low[oi])
            if pl < MIN_P_LOW:
                continue

            ev_low = pl * odds - 1.0
            if ev_low < EV_LOW_THRESHOLD:
                continue

            if best is None or ev_low > best["EV_low"]:
                best = {
                    "Pick": outcome,
                    "Book": book,
                    "BestOdds": float(odds),
                    "p_low": float(pl),
                    "p_med": float(p_med[oi]),
                    "EV_low": float(ev_low),
                }

        if best is None:
            continue

        win = (row["FTR"] == best["Pick"])
        profit = STAKE * (best["BestOdds"] - 1.0) if win else -STAKE
        profits.append(profit)

        picks.append({
            "Season": season,
            "Date": row["Date"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "Pick": best["Pick"],
            "Book": best["Book"],
            "BestOdds": best["BestOdds"],
            "p_low": best["p_low"],
            "p_med": best["p_med"],
            "EV_low": best["EV_low"],
            "Result": row["FTR"],
            "Profit": profit,
        })

    profits = np.asarray(profits, float)
    bets = int(len(profits))
    total_profit = float(profits.sum())
    roi = total_profit / (bets * STAKE) if bets > 0 else 0.0

    picks_df = pd.DataFrame(picks)
    return bets, total_profit, roi, picks_df


def main():
    print("=== Multi-season predictor (shots JSON + rolling within-season) ===")
    print(f"Seasons: {SEASONS}")
    print(f"MIN_TRAIN_MATCHES={MIN_TRAIN_MATCHES} RETRAIN_EVERY={RETRAIN_EVERY} HALF_LIFE_DAYS={HALF_LIFE_DAYS}")
    print(f"N_MODELS={N_MODELS} P_LOW_Q={P_LOW_Q} MIN_P_LOW={MIN_P_LOW} EV_LOW_THRESHOLD={EV_LOW_THRESHOLD}")

    summary = []
    total_bets = 0
    total_profit = 0.0

    for s in SEASONS:
        print(f"\n--- Season {s} ---")
        bets, profit, roi, picks = backtest_season(s)
        print(f"bets={bets:4d} profit={profit:+.3f} ROI={roi:+.2%}")
        summary.append({"season": s, "bets": bets, "profit": profit, "roi": roi})

        total_bets += bets
        total_profit += profit

        if not picks.empty:
            out = f"picks_shots_{s}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            picks.sort_values(["Date", "EV_low"], ascending=[True, False]).to_csv(out, index=False)
            print(f"Saved picks: {out}")

    total_roi = total_profit / (total_bets * STAKE) if total_bets > 0 else 0.0

    res = pd.DataFrame(summary)
    out_sum = f"summary_shots_multiseason_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    res.to_csv(out_sum, index=False)

    print("\n=== Aggregate ===")
    print(f"Total bets={total_bets} total_profit={total_profit:+.3f} ROI={total_roi:+.2%}")
    print(f"Saved summary: {out_sum}")


if __name__ == "__main__":
    main()

