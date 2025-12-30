import json
import time
from pathlib import Path
from typing import Any, Optional, Iterable

from understatapi import UnderstatClient

LEAGUE = "EPL"
SEASONS = list(range(2020, 2025))  # 2016..2024 (completed)
CACHE_DIR = Path("understat_cache")
CACHE_DIR.mkdir(exist_ok=True)

SLEEP_SEC = 0.15  # increase if throttled


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_league_matches(season: int) -> list[dict]:
    out = CACHE_DIR / f"{LEAGUE}_{season}_matches.json"
    if out.exists():
        print(f"[cache] {out.name}")
        return load_json(out)

    print(f"[fetch] league matches {LEAGUE} {season}")
    with UnderstatClient() as us:
        matches = us.league(league=LEAGUE).get_match_data(season=str(season))
    save_json(out, matches)
    print(f"[saved] {out.name} (matches={len(matches)})")
    return matches


def try_get_match_shots(us: UnderstatClient, match_id: str) -> Optional[dict]:
    m = us.match(match=str(match_id))
    for method in ("get_shot_data", "get_shots_data", "get_shots", "get_shots_data_by_match"):
        if hasattr(m, method):
            return getattr(m, method)()
    return None


def cache_match_shots(matches: list[dict]) -> None:
    played = [m for m in matches if m.get("isResult") is True]
    ids = [str(m["id"]) for m in played if m.get("id") is not None]
    print(f"[info] matches total={len(matches)} played={len(played)} ids={len(ids)}")

    with UnderstatClient() as us:
        ok = 0
        exist = 0
        missing_method = 0
        failed = 0

        for i, mid in enumerate(ids, start=1):
            out = CACHE_DIR / f"match_{mid}_shots.json"
            if out.exists():
                exist += 1
                continue

            try:
                shots = try_get_match_shots(us, mid)
                if shots is None:
                    missing_method += 1
                    save_json(out, {"_error": "shots_not_supported_by_this_understatapi_version"})
                else:
                    save_json(out, shots)
                    ok += 1
            except Exception as e:
                failed += 1
                save_json(out, {"_error": f"fetch_failed: {type(e).__name__}: {str(e)}"})

            if i % 50 == 0:
                print(f"[progress] {i}/{len(ids)} ok={ok} exist={exist} missing={missing_method} failed={failed}")

            time.sleep(SLEEP_SEC)

    print(f"[done] ok={ok}, exist={exist}, missing_method={missing_method}, failed={failed}")


def main():
    for s in SEASONS:
        print(f"\n=== Season {s} ===")
        matches = fetch_league_matches(s)
        cache_match_shots(matches)


if __name__ == "__main__":
    main()
