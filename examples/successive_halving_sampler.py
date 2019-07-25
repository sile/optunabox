import optuna
import optunabox


def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_int('y', -5, 5)
    return x**2 + y


if __name__ == '__main__':
    base_sampler = optuna.samplers.TPESampler()
    sampler = optunabox.samplers.SuccessiveHalvingSampler(base_sampler)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)

    print('Best trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print('    {}: {}'.format(key, value))
