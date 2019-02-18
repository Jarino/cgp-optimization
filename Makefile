baseline_eval:
	python -m tengp_eval.optimizers.baseline results/baseline/ hyperparams/baseline/

pso:
	python ccgp_pso.py pso.ini test

pso_hyperopt:
	python -m tengp_eval.optimizers.pso hyperparams/pso/

pso_eval_default:
	python -m tengp_eval.optimizers.runner pso results/pso-default/ hyperparams/pso/ default.ini

sa_eval_default:
	python -m tengp_eval.optimizers.runner sa results/sa-default/ hyperparams/sa/ default.ini

pso_eval:
	python -m tengp_eval.optimizers.runner pso results/pso/ hyperparams/pso/

