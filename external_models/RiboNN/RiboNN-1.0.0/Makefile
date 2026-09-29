install:
	~/miniforge3/bin/mamba env create -f environment.yml

train:
	export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
	python3 -m src.main --train

transfer_learning:
	export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
	python3 -m src.main --transfer_learning

predict:
	python3 -m src.main --predict