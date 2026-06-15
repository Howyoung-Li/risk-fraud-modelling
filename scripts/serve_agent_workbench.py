from __future__ import annotations

from functools import partial
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.agent_workbench import RiskAgent


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"


def serialize_response(response) -> dict:
	return {
		"question": response.question,
		"intent": response.intent,
		"answer": response.answer,
		"tools_called": response.tools_called,
		"evidence": response.evidence,
		"trace": response.trace,
	}


class AgentWorkbenchHandler(SimpleHTTPRequestHandler):
	def do_GET(self) -> None:
		parsed = urlparse(self.path)
		if parsed.path == "/api/agent":
			self.handle_agent_request(parsed.query)
			return
		super().do_GET()

	def handle_agent_request(self, query: str) -> None:
		params = parse_qs(query)
		question = params.get("question", [""])[0].strip()
		if not question:
			self.send_json({"error": "question is required"}, status=400)
			return

		response = RiskAgent().answer(question)
		self.send_json(serialize_response(response))

	def send_json(self, payload: dict, status: int = 200) -> None:
		body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json; charset=utf-8")
		self.send_header("Content-Length", str(len(body)))
		self.end_headers()
		self.wfile.write(body)


def main() -> None:
	handler = partial(AgentWorkbenchHandler, directory=str(DOCS_DIR))
	server = ThreadingHTTPServer(("127.0.0.1", 4173), handler)
	print("Serving Fraud Risk Agent Workbench at http://127.0.0.1:4173/")
	print("Agent API: http://127.0.0.1:4173/api/agent?question=...")
	server.serve_forever()


if __name__ == "__main__":
	main()
