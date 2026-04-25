import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from inference import BotInferenceService


DEFAULT_DATABASE_URL = "postgresql://ddz_dev:password@localhost:5432/ddz?sslmode=disable"


class LazyBotService:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._lock = threading.Lock()
        self._service: Optional[BotInferenceService] = None

    def is_ready(self) -> bool:
        return self._service is not None

    def get_service(self) -> BotInferenceService:
        if self._service is not None:
            return self._service

        with self._lock:
            if self._service is None:
                print("loading inference service")
                self._service = BotInferenceService(self.database_url)
                print("inference service ready")
            return self._service

class BotRequestHandler(BaseHTTPRequestHandler):
    service_holder: LazyBotService

    def do_GET(self) -> None:
        if self.path not in ("/", "/healthz"):
            self._write_json(404, {"error": "not found"})
            return

        self._write_json(200, {"ok": True, "ready": self.service_holder.is_ready()})

    def do_HEAD(self) -> None:
        if self.path not in ("/", "/healthz"):
            self._write_json(404, {"error": "not found"}, head_only=True)
            return

        self._write_json(200, {"ok": True, "ready": self.service_holder.is_ready()}, head_only=True)

    def do_POST(self) -> None:
        if self.path != "/choose-move":
            self._write_json(404, {"error": "not found"})
            return

        try:
            request = self._read_json()
            response = self.service_holder.get_service().choose_move(
                game_id=int(request["game_id"]),
                hand_id=int(request["hand_id"]),
                turn_number=int(request["turn_number"]),
            )
            self._write_json(200, response)
        except KeyError as err:
            self._write_json(400, {"error": f"missing field: {err.args[0]}"})
        except ValueError as err:
            self._write_json(400, {"error": str(err)})
        except Exception as err:
            print(f"request failed: {err}")
            self._write_json(500, {"error": "internal server error"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _write_json(self, status: int, payload: dict[str, Any], head_only: bool = False) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)


def main() -> None:
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    service_holder = LazyBotService(database_url)
    BotRequestHandler.service_holder = service_holder

    server = ThreadingHTTPServer((host, port), BotRequestHandler)
    print(f"bot service listening on {host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
