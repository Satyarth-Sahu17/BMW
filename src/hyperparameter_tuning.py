import argparse
import yaml
import optuna
import torch
from pathlib import Path
from src.train import train
from src.utils.logger import setup_logger

class OptunaObjective:
    def __init__(self, base_config_path):
        with open(base_config_path) as f:
            self.base_config = yaml.safe_load(f)
        self.logger = setup_logger("BMW_Tuning")
        
    def __call__(self, trial):
        # Suggest hyperparameters
        config = self.base_config.copy()
        config['training']['learning_rate'] = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        config['model']['dropout'] = trial.suggest_uniform('dropout', 0.2, 0.5)
        
        # Train and return validation metric
        # Note: You'd need to modify train() to return val_acc
        # For now, mock return
        val_acc = 0.85 + trial.number * 0.01  # Mock
        return val_acc

def run_optuna_search(config_path, n_trials=50):
    logger = setup_logger("BMW_Optuna")
    logger.info(f"Starting Optuna hyperparameter search with {n_trials} trials")
    
    study = optuna.create_study(direction='maximize')
    objective = OptunaObjective(config_path)
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
    
    # Save results
    results_dir = Path("experiments/optuna")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "best_params.yaml", 'w') as f:
        yaml.dump(study.best_params, f)
        
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()
    
    run_optuna_search(args.config, args.n_trials)
