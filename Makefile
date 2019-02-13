pso:
	python ccgp_pso.py pso.ini test

pso_hyperopt:
	python -m tengp_eval.optimizers.pso_hyperopt hyperparams/pso/

pso_eval:
	python -m tengp_eval.optimizers.pso results/pso/ hyperparams/pso/
