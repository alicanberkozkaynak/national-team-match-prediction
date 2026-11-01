from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib

from national_team_score_model import predict_match

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "national_team_model.joblib"
HOST = "127.0.0.1"
PORT = 8000


def load_teams() -> list[str]:
    bundle = joblib.load(MODEL_PATH)
    ratings = bundle.get("state", {}).get("ratings", {})
    fifa = bundle.get("fifa")
    fifa_teams = set(fifa["team"].dropna().astype(str)) if fifa is not None else set()
    return sorted(set(ratings) & fifa_teams, key=str.casefold)


TEAMS = load_teams()


class AppHandler(BaseHTTPRequestHandler):
    server_version = "MilliMacTahmin/1.0"

    def _send_json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file() or STATIC_DIR not in path.resolve().parents:
            self.send_error(404)
            return
        content = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        self.send_response(200)
        self.send_header("Content-Type", (content_type or "application/octet-stream") + ("; charset=utf-8" if path.suffix in {".html", ".css", ".js"} else ""))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/teams":
            self._send_json({"teams": TEAMS, "count": len(TEAMS)})
            return
        if path == "/api/health":
            self._send_json({"status": "ok", "model_exists": MODEL_PATH.exists()})
            return
        if path == "/":
            self._send_file(STATIC_DIR / "index.html")
            return
        requested = (STATIC_DIR / path.lstrip("/")).resolve()
        self._send_file(requested)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            home_team = str(data.get("home_team", "")).strip()
            away_team = str(data.get("away_team", "")).strip()
            tournament = str(data.get("tournament", "Friendly")).strip() or "Friendly"
            neutral = bool(data.get("neutral", False))

            if not home_team or not away_team:
                raise ValueError("İki takım da seçilmeli.")
            if home_team == away_team:
                raise ValueError("Ev sahibi ve deplasman takımı aynı olamaz.")

            result = predict_match(
                home_team=home_team,
                away_team=away_team,
                neutral=neutral,
                tournament=tournament,
                model_path=MODEL_PATH,
            )
            self._send_json(result)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"Tahmin sırasında beklenmeyen hata: {exc}"}, status=500)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Milli Maç Tahmin sitesi: http://{HOST}:{PORT}")
    print("Kapatmak için Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatıldı.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
