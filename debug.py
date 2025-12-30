import requests
from pathlib import Path

def main():
    league = "EPL"
    season = 2024
    url = f"https://understat.com/league/{league}/{season}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://understat.com/",
        "Connection": "keep-alive",
    }

    r = requests.get(url, headers=headers, timeout=30)
    print("STATUS:", r.status_code)
    print("FINAL URL:", r.url)

    Path("understat_cache").mkdir(exist_ok=True)
    html_path = Path("understat_cache") / f"DEBUG_{league}_{season}.html"
    html_path.write_text(r.text, encoding="utf-8")
    print("Saved HTML to:", html_path.resolve())

    text = r.text
    print("\n--- FIRST 500 CHARS ---")
    print(text[:500])
    print("\n--- CONTAINS KEYWORDS ---")
    for k in ["matchesData", "JSON.parse", "cloudflare", "Access denied", "captcha"]:
        print(k, "=>", (k.lower() in text.lower()))

if __name__ == "__main__":
    main()
