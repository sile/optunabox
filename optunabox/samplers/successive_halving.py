from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Union  # NOQA

from optuna import distributions  # NOQA
from optuna.samplers.base import BaseSampler
from optuna.storages.base import BaseStorage  # NOQA
from optuna import structs  # NOQA
from optuna.structs import FrozenTrial  # NOQA
from optuna.structs import StudyDirection
from optuna.study import InTrialStudy  # NOQA


class SuccessiveHalvingSampler(BaseSampler):
    def __init__(self, base_sampler, min_resource=20, reduction_factor=3):
        # type: (BaseSampler, int, int) -> None

        self.base_sampler = base_sampler
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor

    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, distributions.BaseDistribution]

        # TODO: return self.base_sampler

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, distributions.BaseDistribution) -> float

        trials = sorted(storage.get_all_trials(study_id), key=lambda t: t.trial_id)

        # We assume `trials[-1]` is the current trial
        # (this may not be true if you are executing distribution optimization).
        current_trial = trials.pop(-1)

        is_minimize = storage.get_study_direction(study_id) == StudyDirection.MINIMIZE
        sha = SuccessiveHalving(is_minimize, self.min_resource, self.reduction_factor)
        for trial in trials:
            sha.tell(trial)

        trials = sha.active_config.trials
        trials.append(current_trial)

        storage = SuccessiveHalvingStorage(storage, trials)
        return self.base_sampler.sample(storage, study_id, param_name, param_distribution)


class SuccessiveHalving(object):
    def __init__(self, is_minimize, min_resource, reduction_factor):
        # type: (bool, int, int) -> None

        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.is_minimize = is_minimize
        self.active_config = Config(self.min_resource, self.reduction_factor)
        self.pendings = []  # type: List[List[Union[Config, PromotedConfig]]]

    def tell(self, trial):
        # type: (FrozenTrial) -> None

        self._update_active_config()
        self.active_config.tell(trial, self.is_minimize)

    def _update_active_config(self):
        # type: () -> None

        if self.active_config.curr_steps < self.active_config.rung_steps:
            return

        rung_i = self.active_config.rung
        if len(self.pendings) <= rung_i:
            self.pendings.append([])

        rung = self.pendings[rung_i]
        rung.append(self.active_config)
        rung.sort(key=lambda c: c.best_value)
        for i in range(len(rung) // self.reduction_factor):
            if isinstance(rung[i], PromotedConfig):
                continue

            promoted = PromotedConfig(rung[i].best_value)
            self.active_config = rung[i]  # type: ignore
            self.active_config.rung_steps *= self.reduction_factor
            rung[i] = promoted
            return

        self.active_config = Config(self.min_resource, self.reduction_factor)


class PromotedConfig(object):
    def __init__(self, best_value):
        # type: (float) -> None

        self.best_value = best_value


class Config(object):
    def __init__(self, min_resource, reduction_factor):
        # type: (int, int) -> None

        self.min_resource = min_resource
        self.reduction_factor = reduction_factor

        self.rung_steps = min_resource
        self.trials = []  # type: List[FrozenTrial]
        self.best_value = float('inf')

    @property
    def rung(self):
        # type: () -> int

        r = self.min_resource
        i = 0
        while True:
            if self.rung_steps == r:
                return i
            r *= self.reduction_factor
            i += 1

    @property
    def curr_steps(self):
        # type: () -> int

        return len(self.trials)

    def tell(self, trial, is_minimize):
        # type: (FrozenTrial, bool) -> None

        self.trials.append(trial)

        if trial.value is None:
            value = float('inf')
        elif is_minimize:
            value = trial.value
        else:
            value = -trial.value

        self.best_value = min(self.best_value, value)


class SuccessiveHalvingStorage(BaseStorage):
    def __init__(self, inner, trials):
        # type: (BaseStorage, List[FrozenTrial]) -> None

        self.inner = inner
        self.trials = trials

    def create_new_study_id(self, study_name=None):
        # type: (Optional[str]) -> int

        return self.inner.create_new_study_id(study_name)

    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        self.inner.set_study_user_attr(study_id, key, value)

    def set_study_direction(self, study_id, direction):
        # type: (int, StudyDirection) -> None

        self.inner.set_study_direction(study_id, direction)

    def set_study_system_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        self.inner.set_study_system_attr(study_id, key, value)

    def get_study_id_from_name(self, study_name):
        # type: (str) -> int

        return self.inner.get_study_id_from_name(study_name)

    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int

        return self.inner.get_study_id_from_trial_id(trial_id)

    def get_study_name_from_id(self, study_id):
        # type: (int) -> str

        return self.inner.get_study_name_from_id(study_id)

    def get_study_direction(self, study_id):
        # type: (int) -> StudyDirection

        return StudyDirection.MINIMIZE

    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        return self.inner.get_study_user_attrs(study_id)

    def get_study_system_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        return self.inner.get_study_system_attrs(study_id)

    def get_all_study_summaries(self):
        # type: () -> List[structs.StudySummary]

        return self.inner.get_all_study_summaries()

    def create_new_trial_id(self, study_id):
        # type: (int) -> int

        return self.inner.create_new_trial_id(study_id)

    def set_trial_state(self, trial_id, state):
        # type: (int, structs.TrialState) -> None

        self.inner.set_trial_state(trial_id, state)

    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool

        return self.inner.set_trial_param(trial_id, param_name, param_value_internal, distribution)

    def get_trial_number_from_id(self, trial_id):
        # type: (int) -> int

        return self.inner.get_trial_number_from_id(trial_id)

    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float

        return self.inner.get_trial_param(trial_id, param_name)

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        self.inner.set_trial_value(trial_id, value)

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        return self.inner.set_trial_intermediate_value(trial_id, step, intermediate_value)

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        self.inner.set_trial_user_attr(trial_id, key, value)

    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        self.inner.set_trial_system_attr(trial_id, key, value)

    def get_trial(self, trial_id):
        # type: (int) -> structs.FrozenTrial

        return self.inner.get_trial(trial_id)

    def get_all_trials(self, study_id):
        # type: (int) -> List[structs.FrozenTrial]

        return self.trials

    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[structs.TrialState]) -> int

        return len(self.trials)
