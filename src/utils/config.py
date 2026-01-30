from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class SplitConfig:
	method: str
	id_col: str
	time_col: str
	target_col: str
	train_frac: float
	valid_frac: float
	test_frac: float
	seed: int

	# NEW: CV controls
	cv_strategy: str   # "time" or "kfold"
	cv_folds: int
	cv_gap: int


@dataclass(frozen=True)
class ClassImbalanceConfig:
	use_scale_pos_weight: bool
	scale_pos_weight: Optional[float]  # None: compute from train labels.


@dataclass(frozen=True)
class SearchConfig:
	enabled: bool
	method: str                 # "grid" or "random"
	scoring: str
	n_jobs: int
	param_grid: Dict[str, List[Any]]
	n_iter: Optional[int] = None  # used for random search


@dataclass(frozen=True)
class ModelConfig:
	name: str
	params: Dict[str, Any]
	early_stopping_rounds: int
	search: Optional[SearchConfig]  # NEW


@dataclass(frozen=True)
class TorchConfig:
	seed: int
	batch_size: int
	epochs: int
	lr: float
	hidden_sizes: List[int]
	dropout: float


@dataclass(frozen=True)
class Config:
	raw_dir: Path
	processed_dir: Path
	artifacts_dir: Path

	split: SplitConfig
	model_run: List[str]
	evaluation: EvaluationConfig
	class_imbalance: ClassImbalanceConfig
	explain: ExplainConfig

	xgboost: Optional[ModelConfig]
	lightgbm: Optional[ModelConfig]
	#catboost: Optional[ModelConfig]
	# torch: Optional[TorchConfig]

	reduce_memory: bool
	categorical_fill: str
	numerical_fill: str
	use_test_for_encoding: bool

@dataclass(frozen=True)
class EvaluationConfig:
	topk_percents: list[float]
	make_calibration: bool
	calibration_bins: int
	amount_col: str

@dataclass(frozen=True)
class ExplainConfig:
	model_name: str
	shap_sample_size: int
	local_cases: int
	reason_codes_top_n: int
	random_seed: int

def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
	"""Safe nested get: path like 'a.b.c'."""
	cur: Any = d
	for k in path.split("."):
		if not isinstance(cur, dict) or k not in cur:
			return default
		cur = cur[k]
	return cur


def _parse_search(block: Any) -> Optional[SearchConfig]:
	if not isinstance(block, dict):
		return None

	enabled = bool(block.get("enabled", False))
	if not enabled:
		return SearchConfig(
			enabled=False,
			method=str(block.get("method", "grid")),
			scoring=str(block.get("scoring", "roc_auc")),
			n_jobs=int(block.get("n_jobs", -1)),
			param_grid=dict(block.get("param_grid", {})),
			n_iter=block.get("n_iter", None),
		)

	return SearchConfig(
		enabled=True,
		method=str(block.get("method", "grid")),
		scoring=str(block.get("scoring", "roc_auc")),
		n_jobs=int(block.get("n_jobs", -1)),
		param_grid=dict(block.get("param_grid", {})),
		n_iter=(int(block["n_iter"]) if block.get("n_iter") is not None else None),
	)


def load_config(config_path: str = "configs/config.yaml") -> Config:
	with open(config_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	# paths
	raw_dir = Path(_get(cfg, "data.raw_dir", "data/raw"))
	processed_dir = Path(_get(cfg, "data.processed_dir", "data/processed"))
	artifacts_dir = Path(_get(cfg, "artifacts.dir", "artifacts"))

	# split
	split_block = cfg.get("split", {})
	split = SplitConfig(
		method=str(split_block.get("method", "time")),
		id_col=str(split_block.get("id_col", "TransactionID")),
		time_col=str(split_block.get("time_col", "TransactionDT")),
		target_col=str(split_block.get("target_col", "isFraud")),
		train_frac=float(split_block.get("train_frac", 0.7)),
		valid_frac=float(split_block.get("valid_frac", 0.15)),
		test_frac=float(split_block.get("test_frac", 0.15)),
		seed=int(split_block.get("seed", 42)),

		# NEW: CV controls
		cv_strategy=str(split_block.get("cv_strategy", "time")),
		cv_folds=int(split_block.get("cv_folds", 5)),
		cv_gap=int(split_block.get("cv_gap", 0)),
	)

	# model run list (FIX: models.run, not model.run)
	model_run = list(_get(cfg, "models.run", []))
	
	# evaluation
	eval_block = cfg.get("evaluation", {})
	evaluation = EvaluationConfig(
		topk_percents= [float(x) for x in eval_block.get("topk_percents",[0.1, 0.5, 1, 3, 5, 10])],
		make_calibration=bool(eval_block.get("make_calibration", True)),
		calibration_bins=int(eval_block.get("calibration_bins", 20)),
		amount_col=str(eval_block.get("amount_col", "TransactionAmt"))
	)

	# explain
	explain_block = cfg.get("explain",{})
	explain = ExplainConfig(
		model_name= str(explain_block.get("model_name", "lightgbm")),
		shap_sample_size=int(explain_block.get("shap_sample_size", 20000)),
		local_cases= int(explain_block.get("loca_cases", 8)),
		reason_codes_top_n= int(explain_block.get("reason_codes_top_n", 5)),
		random_seed=int(explain_block.get("random_seed", 42))
	)

	# class imbalance
	ci = cfg.get("class_imbalance", {})
	spw = ci.get("scale_pos_weight", None)
	class_imbalance = ClassImbalanceConfig(
		use_scale_pos_weight=bool(ci.get("use_scale_pos_weight", True)),
		scale_pos_weight=(float(spw) if spw is not None else None),
	)

	# xgboost / lightgbm blocks (extended with search)
	xgboost = None
	xgb_block = cfg.get("xgboost")
	if isinstance(xgb_block, dict):
		xgboost = ModelConfig(
			name="xgboost",
			params=dict(xgb_block.get("params", {})),
			early_stopping_rounds=int(xgb_block.get("early_stopping_rounds", 200)),
			search=_parse_search(xgb_block.get("search")),
		)

	lightgbm = None
	lgbm_block = cfg.get("lightgbm")
	if isinstance(lgbm_block, dict):
		lightgbm = ModelConfig(
			name="lightgbm",
			params=dict(lgbm_block.get("params", {})),
			early_stopping_rounds=int(lgbm_block.get("early_stopping_rounds", 200)),
			search=_parse_search(lgbm_block.get("search")),
		)
	"""
	catboost = None
	cat_block = cfg.get("catboost")
	if isinstance(cat_block, dict):
		catboost = ModelConfig(
			name="catboost",
			params=dict(cat_block.get("params", {})),
			early_stopping_rounds=int(cat_block.get("early_stopping_rounds", 200)),
			search=_parse_search(cat_block.get("search")),
		)

	# torch block
	torch = None
	torch_block = cfg.get("torch")
	if isinstance(torch_block, dict):
		torch = TorchConfig(
			seed=int(torch_block.get("seed", 42)),
			batch_size=int(torch_block.get("batch_size", 2048)),
			epochs=int(torch_block.get("epochs", 20)),
			lr=float(torch_block.get("lr", 1e-3)),
			hidden_sizes=list(torch_block.get("hidden_sizes", [256, 128])),
			dropout=float(torch_block.get("dropout", 0.2)),
		)
	"""
	# features
	reduce_memory = bool(_get(cfg, "features.reduce_memory", True))
	categorical_fill = str(_get(cfg, "features.categorical_fill", "missing"))
	numerical_fill = str(_get(cfg, "features.numeric_fill", "median"))
	use_test_for_encoding = bool(_get(cfg, "features.use_test_for_encoding", True))

	return Config(
		raw_dir=raw_dir,
		processed_dir=processed_dir,
		artifacts_dir=artifacts_dir,
		split=split,
		model_run=model_run,
		class_imbalance=class_imbalance,
		xgboost=xgboost,
		lightgbm=lightgbm,
		#catboost=catboost,
		#torch=torch,
		reduce_memory=reduce_memory,
		categorical_fill=categorical_fill,
		numerical_fill=numerical_fill,
		use_test_for_encoding=use_test_for_encoding,
		evaluation=evaluation,
		explain=explain
	)
