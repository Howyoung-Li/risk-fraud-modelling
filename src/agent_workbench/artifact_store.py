from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactStore:
	"""Small read-only facade over generated model, policy, SHAP, and monitoring assets."""

	def __init__(self, root: str | Path = ".") -> None:
		self.root = Path(root)
		self.artifacts_dir = self.root / "artifacts"

	def path(self, *parts: str) -> Path:
		return self.artifacts_dir.joinpath(*parts)

	def read_json(self, *parts: str, default: Any | None = None) -> Any:
		path = self.path(*parts)
		if not path.exists():
			if default is not None:
				return default
			raise FileNotFoundError(path)
		return json.loads(path.read_text(encoding="utf-8"))

	def read_csv(self, *parts: str, required: bool = True) -> pd.DataFrame:
		path = self.path(*parts)
		if not path.exists():
			if required:
				raise FileNotFoundError(path)
			return pd.DataFrame()
		return pd.read_csv(path)

	def evidence_path(self, *parts: str) -> str:
		return str(Path("artifacts").joinpath(*parts))
