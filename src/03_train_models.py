from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .utils.config import load_config
from .utils.io import ensure_dir
from .utils.split import time_split_masks
from .utils.metrics import auc_pr, auc_roc

# Model imports
import lightgbm as lgb
import xgboost as xgb
#from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
#import torch
#import torch.nn as nn
#from torch.utils.data import TensorDataset, DataLoader

def compute_scales_pos_weight(y: ndarray) -> float:
	pos = float((y==1).sum())
	neg = float((y==0).sum())
	return neg/pos if pos > 0 else 1.0

def train_lightgbm(X_train, y_train, X_valid, y_valid, cfg):
	params = dict(cfg.lightgbm.params)
	if cfg.class_imbalance.scale_pos_weight:
		spw = cfg.class_imbalance.scale_pos_weight
		if spw is None:
			spw =compute_scales_pos_weight(y_train)
		params["scale_pos_weight"] = float(spw)
	
	params.setdefault("verbosity", -1)
	params.setdefault("n_jobs", -1)
	params.setdefault("metric", "average_precision")

	dtrain = lgb.Dataset(X_train, label=y_train)
	dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

	model = lgb.train(
		params=params,
		train_set=dtrain,
		num_boost_round=int(params.get("n_estimators",5000)),
		valid_sets=[dvalid],
		valid_names = ["valid"],
		callbacks=[
			lgb.early_stopping(stopping_rounds=cfg.lightgbm.early_stopping_rounds, verbose=False),
			lgb.log_evaluation(period=100),
		]
	)
	return model

def train_xgboost(X_train, y_train, X_valid, y_valid, cfg):
	params = dict(cfg.xgboost.params)
	if cfg.class_imbalance.use_scale_pos_weight:
		spw = cfg.class_imbalance.scale_pos_weight
		if spw is None:
			spw = compute_scales_pos_weight(y_train)
		params["scale_pos_weight"] = float(spw)
	
	params.pop("verbosity", None)
	params.setdefault("n_jobs", -1)

	dtrain = xgb.DMatrix(X_train, label=y_train)
	dvaild = xgb.DMatrix(X_valid,label=y_valid)

	model = xgb.train(
		params=params,
		dtrain=dtrain,
		num_boost_round=int(params.get("n_estimators",5000)),
		evals=[(dvaild,"valid")],
		early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
		verbose_eval=100,
	)
	return model
'''
def train_catboost(X_train, y_train, X_valid, y_valid, cfg):
	params = dict(cfg.catboost.params)
	
	if cfg.class_imbalance.use_scale_pos_weight:
		spw = cfg.class_imbalance.scale_pos_weight
		if spw is None:
			spw = compute_scales_pos_weight(y_train)
		params["scale_pos_weight"] = float(spw)
	
	params.setdefault("verbose", 100)
	params.setdefault("thread_count", -1)
	
	model = CatBoostClassifier(**params)
	
	model.fit(
		X_train, y_train,
		eval_set=(X_valid, y_valid),
		early_stopping_rounds=cfg.catboost.early_stopping_rounds,
		verbose=100,
	)
	
	return model

def train_logit(X_train, y_train, cfg):
	# impute missing values (logit requires no  NaNs)
	X_train_filled = X_train.fillna(X_train.median())

	if cfg.class_imbalance.use_scale_pos_weight:
		spw = cfg.class_imbalance.scale_pos_weight
		if spw is None:
			spw = compute_scales_pos_weight(y_train)
		class_weight = {0: 1.0, 1: spw}
	else:
		class_weight = None

	model = LogisticRegression(
		max_iter=1000,
		solver="lbfgs",
		class_weight=class_weight,
	)
	model.fit(X_train_filled,y_train)
	return model

class MLP(nn.Module):
	def __init__(self, input_dim, hidden_sizes, dropout):
		super().__init__()
		layers = []
		prev = input_dim
		for h in hidden_sizes:
			layers.append(nn.Linear(prev,h))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			prev=h
		layers.append(nn.Linear(prev,1))
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x).squeeze(-1)
	
def train_torch_mlp(X_train, y_train, X_valid, y_valid, cfg):
	torch.manual_seed(cfg.torch.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	X_train_filled = X_train.fillna(X_train.median()).astype(np.float32)
	X_valid_filled = X_valid.fillna(X_train.median()).astype(np.float32)

	X_tr = torch.tensor(X_train_filled.astype(np.float32).values, dtype=torch.float32)
	y_tr = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
	X_val = torch.tensor(X_valid_filled.astype(np.float32).values, dtype=torch.float32)
	y_val = torch.tensor(y_valid.astype(np.float32), dtype=torch.float32)
				
	train_ds = TensorDataset(X_tr, y_tr)
	train_loader = DataLoader(train_ds, batch_size=cfg.torch.batch_size, shuffle=True)

	model = MLP(X_train.shape[1], cfg.torch.hidden_sizes, cfg.torch.dropout).to(device)

	if cfg.class_imbalance.use_scale_pos_weight:
		spw = cfg.class_imbalance.scale_pos_weight
		if spw is None:
			spw = compute_scales_pos_weight(y_train)
		pos_weight = torch.tensor([spw], dtype=torch.float32).to(device)
	else:
		pos_weight=None

	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.torch.lr)

	for epoch in range(cfg.torch.epochs):
		model.train()
		for xb, yb in train_loader:
			xb, yb = xb.to(device), yb.to(device)
			optimizer.zero_grad()
			loss = criterion(model(xb), yb)
			loss.backward()
			# add gradient clipping to prevent NaN
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

	return model, device
'''	
def predict_model(model, X, model_name, device=None):
	if model_name == "lightgbm":
		return model.predict(X, num_iteration=model.best_iteration)
	elif model_name == "xgboost":
		return model.predict(xgb.DMatrix(X))
	'''
	elif model_name == "catboost":
		return model.predict(X, prediction_type="Probability")[:, 1]
	elif model_name == "logit":
		X_filled = X.fillna(X.median())
		return model.predict_proba(X_filled)[:,1]
	elif model_name == "torch_mlp":
		model.eval()
		with torch.no_grad():
			# Use provided train_median for consistent imputation
			X_filled = X.fillna(X.median())
			X_t = torch.tensor(X_filled.astype(np.float32).values, dtype=torch.float32).to(device)
			logits = model(X_t)
			probs = torch.sigmoid(logits).cpu().numpy()
			# Check for NaN in output
			if np.isnan(probs).any():
				print(f"[WARNING] NaN detected in predictions, replacing with 0.5")
				probs = np.nan_to_num(probs, nan=0.5)
			return probs
	'''

def main():
	cfg = load_config()

	processed_dir = cfg.processed_dir
	artifacts_dir = cfg.artifacts_dir
	outputs_dir = artifacts_dir/"outputs"
	models_dir = artifacts_dir/"models"
	metrics_dir = artifacts_dir/"metrics"

	ensure_dir(outputs_dir)
	ensure_dir(models_dir)
	ensure_dir(metrics_dir)

	id_col = cfg.split.id_col
	time_col = cfg.split.time_col
	target_col = cfg.split.target_col

	fe_path = processed_dir/ "model_table_fe.parquet"
	if not fe_path.exists():
		raise FileNotFoundError(f"missing: {fe_path}")
	
	print(f"[train models] loading: {fe_path}")
	df = pd.read_parquet(fe_path)

	train_mask, valid_mask, test_mask = time_split_masks(
		df, time_col=time_col,
		train_frac=cfg.split.train_frac,
		valid_frac=cfg.split.valid_frac
	)

	drop_cols = {id_col, time_col, target_col, "uid_hash"}
	feature_cols = [c for c in df.columns if c not in drop_cols]

	X_train = df.loc[train_mask,feature_cols]
	y_train = df.loc[train_mask, target_col].astype(int).to_numpy()
	X_valid = df.loc[valid_mask, feature_cols]
	y_valid = df.loc[valid_mask, target_col].astype(int).to_numpy()
	X_test = df.loc[test_mask, feature_cols]
	y_test = df.loc[test_mask, target_col].astype(int).to_numpy()

	amt_col = cfg.evaluation.amount_col

	for model_name in cfg.model_run:
		print(f"\n[train models] training: {model_name}")
		if model_name == "lightgbm" and cfg.lightgbm:
			model = train_lightgbm(X_train, y_train,X_valid, y_valid, cfg)
			device=None
		elif model_name == "xgboost" and cfg.xgboost:
			model = train_xgboost(X_train, y_train, X_valid, y_valid, cfg)
			device=None
		else:
			print(f"[train models] skipping {model_name}: not configured")
			continue
		'''
		elif model_name == "catboost" and cfg.catboost:
			model = train_catboost(X_train, y_train, X_valid, y_valid, cfg)
			device = None
		elif model_name == "logit":
			model = train_logit(X_train, y_train, cfg)
			device=None
		elif model_name == "torch_mlp" and cfg.torch:
			model,device = train_torch_mlp(X_train,y_train,X_valid,y_valid,cfg)
		'''

			
		# Predict

		valid_pred = predict_model(model, X_valid, model_name, device)
		test_pred = predict_model(model,X_test,model_name,device)

		# Metrics
		pr_valid = auc_pr(y_valid, valid_pred)
		roc_valid = auc_roc(y_valid, valid_pred)
		pr_test = auc_pr(y_test, test_pred)
		roc_test = auc_roc(y_test,test_pred)

		print(f"[{model_name}] valid PR-AUC={pr_valid:.6f} ROC-AUC={roc_valid:.6f}")
		print(f"[{model_name}] test PR-AUC={pr_test:.6f} ROC-AUC={roc_test:.6f}")

		# save outputs
		suffix = "" if model_name=="lightgbm" else f"_{model_name}"

		out_valid = df.loc[valid_mask,[id_col, time_col, target_col]].copy()
		out_valid["score"] = valid_pred.astype("float32")
		out_valid.to_csv(outputs_dir/f"scored_valid{suffix}.csv", index=False)

		out_test = df.loc[test_mask,[id_col, time_col, target_col]].copy()
		out_test["score"] = test_pred.astype("float32")
		out_test.to_csv(outputs_dir/f"scored_test{suffix}.csv", index=False)

		# save model
		if model_name =="lightgbm":
			model_path = models_dir/"lgbm_model.txt"
			model.save_model(str(model_path))
			print(f"[{model_name}] saved model: {model_path}")
		elif model_name =="xgboost":
			model_path = models_dir/"xgb_model.json"
			model.save_model(str(model_path))
			print(f"[{model_name}] saved model: {model_path}")

		# save metrics
		metrics = {
			"valid_pr_auc": pr_valid,
			"valid_roc_auc": roc_valid,
			"test_pr_auc": pr_test,
			"test_roc_auc": roc_test,
		}
		with open(metrics_dir/f"{model_name}_summary.json","w") as f:
			json.dump(metrics,f,indent=2)

		print(f"[{model_name}] saved outputs and metrics")
	
if __name__ == "__main__":
	main()

				