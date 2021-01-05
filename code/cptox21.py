# This script is part of the supporting information to the manuscript entitled
# "Assessing the Calibration in Toxicological in Vitro Models with Conformal Prediction".
# The script was developed by Andrea Morger in the In Silico Toxicology and Structural Biology Group of
# Prof. Dr. Andrea Volkamer at the Charité Universitätsmedizin Berlin, in collaboration with
# Fredrik Svensson, Staffan Arvidsson Mc Shane, Niharika Gauraha, Ulf Norinder and Ola Spjuth.
# It was last updated in December 2020.

import pandas as pd
import numpy as np
import random
import os
import math

import copy
import matplotlib.pyplot as plt
import scipy

from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from nonconformist.icp import IcpClassifier

import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Load/handle signatures
# -------------------------------------------------------------------


def define_path(endpoint, data, signatures_path):
    """
    Define the path where the signatures are stored.
    As they were all created with cpsign, the path is similar
    """
    path = os.path.join(
        signatures_path, f"models_{endpoint}_{data}/sparse_data/data.csr"
    )
    return path


def load_signatures_files(path1, path2, path3):
    """
    Load signatures from multiple .csr files (for multiple datasets)
    This has the advantage, that the length of the signatures is automatically padded

    Parameters
    ----------
    path1 : Path to dataset 1, e.g. Tox21train
    path2 : Path to dataset 2, e.g. Tox21test
    path3 : Path to dataset 3, e.g. Tox21score

    Returns
    -------
    X and y arrays for the three datasets
    """
    # fixme: this function might be adapted to accept any number of paths
    X1, y1, X2, y2, X3, y3 = load_svmlight_files([path1, path2, path3])
    return X1, y1, X2, y2, X3, y3


def combine_csr(X1, y1, X2, y2):
    """
    A function that combines two sparse matrices (signatures and labels). This is e.g. used for train_update in CPTox21
    """
    X1_coo = X1.tocoo()
    X2_coo = X2.tocoo()

    len_X1 = X1_coo.shape[0]
    X2_coo.row = np.array([i + len_X1 for i in X2_coo.row])

    coo_data = scipy.concatenate((X1_coo.data, X2_coo.data))
    coo_rows = scipy.concatenate((X1_coo.row, X2_coo.row))
    coo_cols = scipy.concatenate((X1_coo.col, X2_coo.col))

    X_comb_coo = scipy.sparse.coo_matrix(
        (coo_data, (coo_rows, coo_cols)),
        shape=(X1_coo.shape[0] + X2_coo.shape[0], X1_coo.shape[1]),
    )
    X_comb = X_comb_coo.tocsr()
    y_comb = np.append(y1, y2, axis=0)
    return X_comb, y_comb


# --------------------------------
# Samplers
# --------------------------------


class Sampler:
    """
    Basic 'sampler' class, to generate samples/subsets for the different conformal prediction steps
    """

    def _gen_samples(self, y):
        raise NotImplementedError("Implement in your subclass")
        pass

    def gen_samples(self, labels):
        """

        Parameters
        ----------
        labels : pd.Series
            a series of labels for the molecules

        Returns
        -------

        """
        y = labels
        return self._gen_samples(y)

    @staticmethod
    def _balance(y_idx, idx, ratio=1.0):
        # Mask to distinguish compounds of inactive and active class of dataset
        mask_0 = y_idx == 0
        y_0 = idx[mask_0]
        mask_1 = y_idx == 1
        y_1 = idx[mask_1]

        # Define which class corresponds to larger proper training set and is subject to undersampling
        larger = y_0 if y_0.size > y_1.size else y_1
        smaller = y_1 if y_0.size > y_1.size else y_0

        # Subsample larger class until same number of instances as for smaller class is reached
        while smaller.size < larger.size / ratio:
            k = np.random.choice(range(larger.size))
            larger = np.delete(larger, k)

            idx = sorted(np.append(larger, smaller))
        assert len(idx) == 2 * len(smaller)

        return idx

    @property
    def name(self):
        raise NotImplementedError("Implement in your subclass")


class CrossValidationSampler(Sampler):
    """
    This is a sampler to be used for crossvalidation or cross-conformal predictors (not implemented yet)

    Parameters
    ----------
    n_folds : int
        Number of folds. Must be at least 2

    Attributes
    ----------
    n_folds : int
        Number of folds. Must be at least 2

    Examples
    --------
    todo
    """

    def __init__(self, n_folds=5, random_state=None):
        self.n_folds = n_folds
        self.random_state = random_state

    def _gen_samples(self, y):
        folds = StratifiedKFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
        for i, (train, test) in enumerate(folds.split(X=np.zeros(len(y)), y=y)):
            # i specifies the fold of the crossvalidation, i.e. between 0 and 4
            yield i, train, test

    @property
    def name(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__} with {self.n_folds} folds>"


class StratifiedRatioSampler(Sampler):
    """
    This sampler can e.g. be used for aggregated conformal predictors

    Parameters
    ----------
    test_ratio : float
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
        Note: according to sklearn, test_ratio could also be int or None.
    n_folds : int
        Number of re-shuffling and splitting iterations.

    Attributes
    ----------
    test_ratio : float
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
        Note: according to sklearn, test_ratio could also be int or None.
    n_folds : int
        Number of re-shuffling and splitting iterations.

    Examples
    --------
    todo
    """

    def __init__(self, test_ratio=0.3, n_folds=1, random_state=None):
        self.test_ratio = test_ratio
        self.n_folds = n_folds
        self.random_state = random_state

    def _gen_samples(self, y):
        sss = StratifiedShuffleSplit(n_splits=self.n_folds, test_size=self.test_ratio, random_state=self.random_state)
        for i, (train, test) in enumerate(
            sss.split(X=np.zeros(len(y)), y=y)
        ):  # np.zeros used as a placeholder for X
            yield i, train, test

    @property
    def name(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__} with {self.n_folds} folds and using test_ratio {self.test_ratio}>"


class KnownIndicesSampler(Sampler):
    """
    A sampler which already knows the indices for splitting
    """

    def __init__(self, known_train, known_test):
        known_train_test = []
        for indices in zip(known_train, known_test):
            known_train_test.append(indices)
        self.known_indices = known_train_test

    def _gen_samples(self, y):
        for i, indices in enumerate(self.known_indices):
            train = indices[0]
            test = indices[1]
            yield i, train, test

    @property
    def name(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.known_indices)} folds>"
        # fixme: check if len(self.known_indices) makes sense


# -------------------------------------------------------------------
# Inductive Conformal Predictor
# -------------------------------------------------------------------


class InductiveConformalPredictor(IcpClassifier):
    """
    Inductive Conformal Prediction Classifier
    This is a subclass of the IcpClassifier from nonconformist
    https://github.com/donlnz/nonconformist/blob/master/nonconformist/icp.py
    The subclass allows to further extend the class to the needs of this project

    Parameters
    ----------
    # Note: some of the parameters descriptions are copied from nonconformist IcpClassifier

    condition: condition for calculating p-values. Default condition is mondrian (calibration with 1 list
         of nc scores per class).      Note that default condition in nonconformist is 'lambda x: 0'
         (only one list for both/multiple classes (?)).
         For mondrian condition, see: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00159
    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity of
        calibration examples and test patterns. Should implement ``fit(x, y)``
        and ``calc_nc(x, y)``.

    Attributes
    ----------
    # Note: some of the attributes descriptions are copied from nonconformist IcpClassifier
    condition: condition for calculating p-values. Note that if we want to use 'mondrian' condition,
        we can either input condition='mondrian' or condition=(lambda instance: instance[1]).
        Then, the condition.name will be saved, which is useful for serialisation
    cal_x : numpy array of shape [n_cal_examples, n_features]
        Inputs of calibration set.
    cal_y : numpy array of shape [n_cal_examples]
        Outputs of calibration set.
    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity scores.
    classes : numpy array of shape [n_classes]
        List of class labels, with indices corresponding to output columns
        of IcpClassifier.predict()

    Examples
    --------
    todo
    """

    def __init__(self, nc_function, condition=None, smoothing=False):
        super().__init__(nc_function, condition=condition, smoothing=smoothing)

        # fixme: this subclass was originally there to allow serialisation of the conformal predictors. However,
        #  this is not available yet


# -------------------------------
# Conformal Predictor Aggregators
# -------------------------------


class BaseConformalPredictorAggregator:
    """
    Combines multiple InductiveConformalPredictor predictors into an aggregated model
    The structure of this class is adapted from the nonconformist acp module:
    https://github.com/donlnz/nonconformist/blob/master/nonconformist/acp.py

    Parameters
    ----------
    predictor : object
        Prototype conformal predictor (i.e. InductiveConformalPredictor)
        used for defining conformal predictors included in the aggregate model.

    Attributes
    ----------
    predictor : object
        Prototype conformal predictor (i.e. InductiveConformalPredictor)
        used for defining conformal predictors included in the aggregate model.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def _fit_calibrate(self, **kwargs):
        raise NotImplementedError("Implement in your subclass")

    def fit_calibrate(self, **kwargs):
        return self._fit_calibrate(**kwargs)

    def _predict(self, **kwargs):
        raise NotImplementedError("Implement in your subclass")

    def predict(self, **kwargs):
        return self._predict(**kwargs)

    @property
    def name(self):
        raise NotImplementedError("Implement in your subclass")


class AggregatedConformalPredictor(BaseConformalPredictorAggregator):
    """
    Generates an aggregated conformal predictor (acp) from multiple InductiveConformalPredictor predictors
    The structure of this class is adapted from the nonconformist acp module:
    https://github.com/donlnz/nonconformist/blob/master/nonconformist/acp.py

    Parameters
    ----------
    predictor : object
        Prototype conformal predictor (i.e. InductiveConformalPredictor)
        used for defining conformal predictors included in the aggregate model.
    sampler : object
        Sampler object used to generate training and calibration examples
        for the underlying conformal predictors.
    aggregation_func : callable
        Function used to aggregate the predictions of the underlying
        conformal predictors. Defaults to ``numpy.median``.
    n_models : int
        Number of models to aggregate.

    Attributes
    ----------
    predictor : object
        Prototype conformal predictor (i.e. InductiveConformalPredictor)
        used for defining conformal predictors included in the aggregate model.
    sampler : object
        Sampler object used to generate training and calibration examples
        for the underlying conformal predictors.
    agg_func : callable
        Function used to aggregate the predictions of the underlying
        conformal predictors. Defaults to ``numpy.median``.
    n_models : int
        Number of models to aggregate.
    predictors_fitted : list
        contains fitted ICP's
    predictors_calibrated : list
        contains calibrated ICP's
    predictors_calibrated_update : list
        contains fitted ICP's calibrated with the update dataset

    Examples
    --------
    todo
    """

    def __init__(self, predictor, sampler, aggregation_func=None):
        super().__init__(predictor)
        self.predictor = predictor
        self.predictors_fitted = []
        self.predictors_calibrated = []

        self.sampler = sampler
        self.n_models = sampler.n_folds
        self.agg_func = aggregation_func

    @staticmethod
    def _f(predictor, X):
        return predictor.predict(X, None)

    @staticmethod
    def _f_nc(predictor, X, y):
        pred_proba = predictor.nc_function.model.model.predict_proba(X)
        nc = predictor.nc_function.err_func.apply(pred_proba, y)
        nc_0 = nc[y == 0]
        nc_1 = nc[y == 1]
        return nc_0, nc_1

    def _fit_calibrate(
        self, X_train=None, y_train=None,
    ):

        self.predictors_fitted.clear()
        self.predictors_calibrated.clear()

        samples = self.sampler.gen_samples(labels=y_train)
        for loop, p_train, cal in samples:
            predictor = copy.deepcopy(self.predictor)

            # Fit
            predictor.train_index = p_train
            predictor.fit(X_train[p_train, :], y_train[p_train])
            self.predictors_fitted.append(predictor)

            # Calibrate
            predictor_calibration = copy.deepcopy(predictor)
            predictor_calibration.calibrate(X_train[cal, :], y_train[cal])
            self.predictors_calibrated.append(predictor_calibration)

    def _predict(self, X_score=None):
        predictions = np.dstack(
            [self._f(p, X_score) for p in self.predictors_calibrated]
        )
        predictions = self.agg_func(predictions, axis=2)

        return predictions

    def predict_nc(self, X_score=None, y_score=None):
        nc_0_predictions = [
            self._f_nc(p, X_score, y_score)[0] for p in self.predictors_fitted
        ]
        nc_1_predictions = [
            self._f_nc(p, X_score, y_score)[1] for p in self.predictors_fitted
        ]
        nc_0_predictions = np.concatenate(nc_0_predictions).ravel().tolist()
        nc_1_predictions = np.concatenate(nc_1_predictions).ravel().tolist()
        return nc_0_predictions, nc_1_predictions

    @property
    def name(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__}, samples generated with {self.sampler}, {self.n_models} models built>"


class CPTox21AggregatedConformalPredictor(AggregatedConformalPredictor):
    """
    An aggregated conformal predictor class, specificly adapted for the CPTox21 calupdate part
    """

    def __init__(self, predictor, sampler, aggregation_func=None):
        super().__init__(predictor, sampler, aggregation_func)
        self.predictors_calibrated_update = []
        self.predictors_calibrated_update2 = []

    def _fit_calibrate(
        self,
        X_train=None,
        y_train=None,
        X_update=None,
        y_update=None,
        X_update2=None,
        y_update2=None,
    ):

        self.predictors_fitted.clear()
        self.predictors_calibrated.clear()
        self.predictors_calibrated_update.clear()
        self.predictors_calibrated_update2.clear()

        samples = self.sampler.gen_samples(labels=y_train)

        for loop, p_train, cal in samples:  # i.e. 20 loops
            predictor = copy.deepcopy(self.predictor)

            # Fit
            predictor.train_index = p_train
            predictor.fit(X_train[p_train, :], y_train[p_train])
            self.predictors_fitted.append(predictor)

            # Calibrate
            predictor_calibration = copy.deepcopy(predictor)
            predictor_calibration.calibrate(X_train[cal, :], y_train[cal])
            self.predictors_calibrated.append(predictor_calibration)

            # cal_update - calibrate with "newer" calibration set
            predictor_calibration_update = copy.deepcopy(predictor)
            predictor_calibration_update.calibrate(X_update, y_update)
            self.predictors_calibrated_update.append(predictor_calibration_update)

            predictor_calibration_update2 = copy.deepcopy(predictor)
            predictor_calibration_update2.calibrate(X_update2, y_update2)
            self.predictors_calibrated_update2.append(predictor_calibration_update2)

    def predict_cal_update(self, X_score=None):
        predictions_cal_update = np.dstack(
            [self._f(p, X_score) for p in self.predictors_calibrated_update]
        )
        predictions_cal_update = self.agg_func(predictions_cal_update, axis=2)

        return predictions_cal_update

    def predict_cal_update2(self, X_score=None):
        predictions_cal_update2 = np.dstack(
            [self._f(p, X_score) for p in self.predictors_calibrated_update2]
        )
        predictions_cal_update2 = self.agg_func(predictions_cal_update2, axis=2)
        return predictions_cal_update2


# --------------------------------
# Crossvalidation
# --------------------------------


class CrossValidator:
    """
    This is a class to perform a crossvalidation using aggregated conformal predictors.
    Note that this class only provides predictions within the crossvalidation, i.e.
    of the test set split from X/y. If you want to predict external data within the
    crossvalidation, use one of the provided subclasses or implement your own subclass
    """

    def __init__(self, predictor, cv_splitter):
        self.sampler = cv_splitter
        self.predictor = predictor
        self._evaluation_df_cv = None
        self._cv_predictions = None
        self.cv_predictors = None
        self.num_actives = 0
        self.num_inactives = 0

    def cross_validate(
        self, steps, endpoint=None, X=None, y=None, class_wise_evaluation=False,
    ):

        num_actives = y.sum()
        self.num_actives = num_actives
        self.num_inactives = len(y) - num_actives

        cv_predictions = []
        cv_y_test = []
        cv_predictors = []

        cv_evaluations = self._create_empty_evaluations_dict()

        samples = self.sampler.gen_samples(labels=y)

        for fold, train, test in samples:
            cv_y_test.append(y[test])

            predictor = copy.deepcopy(self.predictor)

            # Fit ACP
            predictor.fit_calibrate(X_train=X[train], y_train=y[train])
            cv_predictors.append(predictor)
            cv_prediction = predictor.predict(X_score=X[test])
            cv_predictions.append(cv_prediction)

            cv_evaluations = self._evaluate(
                cv_prediction,
                y[test],
                cv_evaluations,
                endpoint,
                fold=fold,
                steps=steps,
                class_wise=class_wise_evaluation,
            )

        self._evaluation_df_cv = pd.DataFrame(cv_evaluations)
        self._cv_predictions = [cv_predictions, cv_y_test]
        self.cv_predictors = cv_predictors
        return pd.DataFrame(cv_evaluations)

    @staticmethod
    def _create_empty_evaluations_dict():

        evaluation_measures = [
            "validity",
            "validity_0",
            "validity_1",
            "error_rate",
            "error_rate_0",
            "error_rate_1",
            "efficiency",
            "efficiency_0",
            "efficiency_1",
            "accuracy",
            "accuracy_0",
            "accuracy_1",
        ]

        empty_evaluations_dict = {}
        for measure in evaluation_measures:
            empty_evaluations_dict[measure] = []

        empty_evaluations_dict["significance_level"] = []
        empty_evaluations_dict["fold"] = []

        return empty_evaluations_dict

    @staticmethod
    def _evaluate(
        prediction, y_true, evaluations, endpoint, fold, steps, class_wise=True
    ):
        # fixme later 1: currently class-wise evaluation measures are calculated anyways but only saved
        #  if class_wise is True. Library might be changed, so that they are only calculated if necessary
        # fixme later 2: validity and error_rate could be calculated using the same method, no need to do this twice
        evaluator = Evaluator(prediction, y_true, endpoint)
        sl = [i / float(steps) for i in range(steps)] + [1]

        validities_list = ["validity", "validity_0", "validity_1"]
        error_rates_list = ["error_rate", "error_rate_0", "error_rate_1"]
        efficiencies_list = ["efficiency", "efficiency_0", "efficiency_1"]
        accuracies_list = ["accuracy", "accuracy_0", "accuracy_1"]

        validities = [
            evaluator.calculate_validity(i / float(steps)) for i in range(steps)
        ] + [evaluator.calculate_validity(1)]
        for validity in validities_list:
            evaluations[validity].extend([val[validity] for val in validities])

        error_rates = [
            evaluator.calculate_error_rate(i / float(steps)) for i in range(steps)
        ] + [evaluator.calculate_error_rate(1)]
        for error_rate in error_rates_list:
            evaluations[error_rate].extend([err[error_rate] for err in error_rates])

        efficiencies = [
            evaluator.calculate_efficiency(i / float(steps)) for i in range(steps)
        ] + [evaluator.calculate_efficiency(1)]
        for efficiency in efficiencies_list:
            evaluations[efficiency].extend([eff[efficiency] for eff in efficiencies])
        accuracies = [
            evaluator.calculate_accuracy(i / float(steps)) for i in range(steps)
        ] + [evaluator.calculate_accuracy(1)]
        for accuracy in accuracies_list:
            evaluations[accuracy].extend([acc[accuracy] for acc in accuracies])

        evaluations["significance_level"].extend(sl)
        evaluations["fold"].extend([fold] * (steps + 1))
        return evaluations

    @property
    def averaged_evaluation_df_cv(self):
        return self._average_evaluation_df(
            self._evaluation_df_cv, self.num_actives, self.num_inactives
        )

    @staticmethod
    def _average_evaluation_df(evaluation_df, num_actives, num_inactives):
        evaluation_df_grouped = evaluation_df.groupby(
            by="significance_level"
        ).aggregate([np.mean, np.std])
        evaluation_df_grouped.drop(["fold"], axis=1, inplace=True)
        evaluation_df_grouped.columns = [
            " ".join((a, b)) for a, b in evaluation_df_grouped.columns
        ]
        evaluation_df_grouped.columns = evaluation_df_grouped.columns.get_level_values(
            0
        )
        evaluation_df_grouped["significance_level"] = evaluation_df_grouped.index
        evaluation_df_grouped["num_actives"] = num_actives
        evaluation_df_grouped["num_inactives"] = num_inactives
        return evaluation_df_grouped

    @property
    def cv_predictions_df(self):
        return self._format_predictions_df(self._cv_predictions, self._cv_names)

    @staticmethod
    def _format_predictions_df(predictions, names):
        #         print("names", type(names), names)
        pred_dfs = []
        for i, pred in enumerate(predictions):
            pred_df = pd.DataFrame(data=predictions[0][i])
            pred_df["true"] = predictions[1][i]
            if names is not None:
                pred_df["Name"] = names
            pred_dfs.append(pred_df)

    def calibration_plot(
        self,
        endpoint,
        colours=("blue", "darkred", "deepskyblue", "lightcoral"),
        class_wise=True,
        efficiency=True,
            title_name=None
    ):
        return self._calibration_plot(
            averaged_evaluation_df=self.averaged_evaluation_df_cv,
            endpoint=endpoint,
            colours=colours,
            class_wise=class_wise,
            efficiency=efficiency,
            title_name=title_name
        )

    @staticmethod
    def _calibration_plot(
        averaged_evaluation_df,
        endpoint,
        colours=("blue", "darkred", "deepskyblue", "lightcoral"),
        class_wise=True,
        efficiency=True,
        title_name=None,
    ):

        if class_wise and efficiency:
            evaluation_measures = [
                "error_rate_0",
                "error_rate_1",
                "efficiency_0",
                "efficiency_1",
            ]
        elif class_wise and not efficiency:
            evaluation_measures = ["error_rate_0", "error_rate_1"]

        elif not class_wise and efficiency:
            evaluation_measures = ["error_rate", "efficiency"]

        else:  # not class_wise and not efficiency
            evaluation_measures = ["error_rate"]

        fig, ax = plt.subplots()

        ax.plot([0, 1], [0, 1], "--", linewidth=1, color="black")
        sl = averaged_evaluation_df["significance_level"]

        for ev, colour in zip(evaluation_measures, colours):
            ev_mean = averaged_evaluation_df[f"{ev} mean"]
            ev_std = averaged_evaluation_df[f"{ev} std"]
            ax.plot(sl, ev_mean, label=True, c=colour)
            ax.fill_between(
                sl, ev_mean - ev_std, ev_mean + ev_std, alpha=0.3, color=colour
            )

        major_ticks = np.arange(0, 101, 20)
        minor_ticks = np.arange(0, 101, 5)

        ax.set_xticks(minor_ticks / 100.0, minor=True)
        ax.set_yticks(major_ticks / 100.0)
        ax.set_yticks(minor_ticks / 100.0, minor=True)

        ax.grid(which="minor", linewidth=0.5)
        ax.grid(which="major", linewidth=1.5)

        ax.set_xlabel("significance",)
        ax.set_ylabel("error rate")
        eval_legend = evaluation_measures.copy()
        eval_legend.insert(0, "expected_error_rate")
#         fig.legend(eval_legend, bbox_to_anchor=(1.25, 0.75))
        lgd = ax.legend(eval_legend, loc='center left', bbox_to_anchor=(1, 0.5))
        if title_name is not None:
            plt.title(f"{title_name} - {endpoint}")
        else:
            plt.title(endpoint)
        return plt, lgd


class PredictCrossValidator(CrossValidator):
    def __init__(self, predictor, cv_splitter):
        super().__init__(predictor, cv_splitter)
        self._pred_predictions = None

    def predict(
        self, X_predict=None, y_predict=None,
    ):

        assert self.cv_predictors is not None

        pred_predictions = []

        assert (
            len(self.cv_predictors) == 5
        )  # This assertion is here to assert that the code does what it is thought to do.
        # It can be deleted later (or move to pytest)

        for predictor in self.cv_predictors:
            prediction = predictor.predict(X_score=X_predict)

            pred_predictions.append(prediction)

        pred_predictions = np.mean(pred_predictions, axis=0)

        if y_predict is not None:
            self._pred_predictions = [pred_predictions, y_predict]
        else:
            self._pred_predictions = pred_predictions

        return pred_predictions


class CPTox21CrossValidator(CrossValidator):
    """
    A crossvalidator specifically designed for the CPTox21 calupdate part. Due to memory issues, not all the CPTox21
    experiments can be performed within this crossvalidator class.
    """

    def __init__(self, predictor_acp, cv_splitter, score_splitter):
        super().__init__(predictor_acp, cv_splitter)
        self.cal_update_sampler = score_splitter
        #         self._evaluation_df_cv = None  # already initialised in parent class

        self._evaluation_df_pred_score = None
        self._evaluation_df_pred_test = None
        self._evaluation_df_cal_update = None
        self._evaluation_df_cal_update2 = None
        #         self._cv_predictions = None  # already initialised in parent class

        self._pred_score_predictions = None
        self._pred_test_predictions = None
        self._cal_update_predictions = None
        self._cal_update2_predictions = None

        self._train_ncs = {"nc_0": [], "nc_1": []}
        self._update_ncs = {"nc_0": [], "nc_1": []}
        self._score_ncs = {"nc_0": [], "nc_1": []}

        self._names_predict = None

        self.train_indices = []
        self.test_indices = []

        self.num_actives = 0
        self.num_inactives = 0

    def cross_validate(
        self,
        steps,
        endpoint=None,
        X_train=None,
        y_train=None,
        X_update=None,
        y_update=None,
        X_score=None,
        y_score=None,
        class_wise_evaluation=True,
    ):

        num_actives = y_train.sum()
        self.num_actives = num_actives
        self.num_inactives = len(y_train) - num_actives

        cv_predictions = []
        pred_score_predictions = []
        pred_test_predictions = []
        cal_update_predictions = []
        cal_update2_predictions = []

        cv_predictors = []

        cv_y_test = []
        cv_names_test = []

        cv_evaluations = self._create_empty_evaluations_dict()
        pred_score_evaluations = self._create_empty_evaluations_dict()
        pred_test_evaluations = self._create_empty_evaluations_dict()
        cal_update_evaluations = self._create_empty_evaluations_dict()
        cal_update2_evaluations = self._create_empty_evaluations_dict()

        samples = self.sampler.gen_samples(labels=y_train)

        for fold, train, test in samples:
            self.train_indices.append(list(train))
            self.test_indices.append(list(test))
            cv_y_test.append(y_train[test])

            score_samples = self.cal_update_sampler.gen_samples(labels=y_score)
            for loop, score_cal, score_pred in score_samples:
                # This 'print' is necessary to actually get score_cal and score_pred
                logger.info("len score, calibration:", len(score_cal), "prediction: ", len(score_pred))

            # ----------------------------------------------------------
            # Fit and calibrate ACP
            # ----------------------------------------------------------

            predictor_acp = copy.deepcopy(self.predictor)
            # Fit ACP, both with and without updated calibration set
            logger.info("fitting and calibrating ACP")
            predictor_acp.fit_calibrate(
                X_train=X_train[train],
                y_train=y_train[train],
                X_update=X_update,
                y_update=y_update,
                X_update2=X_score[score_cal],
                y_update2=y_score[score_cal],
            )
            cv_predictors.append(predictor_acp)

            # ----------------------------------------------------------
            # Make predictions with ACP
            # ----------------------------------------------------------

            # CV prediction (internal CV test set)
            logger.info("crossvalidation prediction with original calibration set")
            cv_prediction = predictor_acp.predict(X_score=X_train[test])
            cv_predictions.append(cv_prediction)

            # Predict (external) score set using predictor with and without updated calibration set
            logger.info("predict external data with original calibration set")
            pred_score_prediction = predictor_acp.predict(X_score=X_score)
            pred_test_prediction = predictor_acp.predict(X_score=X_update)
            logger.info("predict external data with updated calibration set")
            cal_update_prediction = predictor_acp.predict_cal_update(
                X_score=X_score
            )
            logger.info(
                "predict part of external data with model calibrated with (other) part of external data"
            )
            cal_update2_prediction = predictor_acp.predict_cal_update2(
                X_score=X_score[score_pred]
            )

            pred_score_predictions.append(pred_score_prediction)
            pred_test_predictions.append(pred_test_prediction)
            cal_update_predictions.append(cal_update_prediction)
            cal_update2_predictions.append(cal_update2_prediction)

            # ----------------------------------------------------------
            # Predict nonconformity scores with ACP
            # ----------------------------------------------------------

            train_test_nc = predictor_acp.predict_nc(
                X_score=X_train[test], y_score=y_train[test]
            )
            self._train_ncs["nc_0"].append(train_test_nc[0])
            self._train_ncs["nc_1"].append(train_test_nc[1])
            update_nc = predictor_acp.predict_nc(X_score=X_update, y_score=y_update)
            self._update_ncs["nc_0"].append(update_nc[0])
            self._update_ncs["nc_1"].append(update_nc[1])
            score_nc = predictor_acp.predict_nc(X_score=X_score, y_score=y_score)
            self._score_ncs["nc_0"].append(score_nc[0])
            self._score_ncs["nc_1"].append(score_nc[1])

            # ----------------------------------------------------------
            # Evaluate predictions
            # ----------------------------------------------------------

            for prediction, y_true, evaluations in zip(
                [
                    cv_prediction,
                    pred_score_prediction,
                    pred_test_prediction,
                    cal_update_prediction,
                    cal_update2_prediction,
                ],
                [y_train[test], y_score, y_update, y_score, y_score[score_pred]],
                [
                    cv_evaluations,
                    pred_score_evaluations,
                    pred_test_evaluations,
                    cal_update_evaluations,
                    cal_update2_evaluations,
                ],
            ):
                # fixme: do we really need evaluations???
                evaluations = self._evaluate(
                    prediction,
                    y_true,
                    evaluations,
                    endpoint=endpoint,
                    fold=fold,
                    steps=steps,
                    class_wise=class_wise_evaluation,
                )

        self._evaluation_df_cv = pd.DataFrame(cv_evaluations)
        self._evaluation_df_pred_score = pd.DataFrame(pred_score_evaluations)
        self._evaluation_df_pred_test = pd.DataFrame(pred_test_evaluations)
        self._evaluation_df_cal_update = pd.DataFrame(cal_update_evaluations)
        self._evaluation_df_cal_update2 = pd.DataFrame(
            cal_update2_evaluations
        )
        self._cv_predictions = [cv_predictions, cv_y_test]
        self._pred_score_predictions = [pred_score_predictions, y_score]
        self._pred_test_predictions = [pred_test_predictions, y_update]
        self._cal_update_predictions = [cal_update_predictions, y_score]
        self._cal_update2_predictions = [
            cal_update2_predictions,
            y_score[score_pred],
        ]
        self.cv_predictors = cv_predictors

        return (
            pd.DataFrame(cv_evaluations),
            pd.DataFrame(pred_score_evaluations),
            pd.DataFrame(pred_test_evaluations),
            pd.DataFrame(cal_update_evaluations),
            pd.DataFrame(cal_update2_evaluations),
        )

    def calibration_plot(
        self,
        averaged_evaluation_df,
        endpoint=None,
        colours=("blue", "darkred", "deepskyblue", "lightcoral"),
        class_wise=True,
        efficiency=True,
        title_name=None,
        **kwargs,
    ):

        return self._calibration_plot(
            averaged_evaluation_df=averaged_evaluation_df,
            endpoint=endpoint,
            colours=colours,
            class_wise=class_wise,
            efficiency=efficiency,
            title_name=title_name,
        )

    def plot_nonconformity_scores(
        self, cl, endpoint, nbins=50, colours=("blue", "orange", "green")
    ):
        train_ncs = self._train_ncs[f"nc_{cl}"]
        train_ncs = [i for sl in train_ncs for i in sl]

        update_ncs = self._update_ncs[f"nc_{cl}"]
        update_ncs = [i for sl in update_ncs for i in sl]

        score_ncs = self._score_ncs[f"nc_{cl}"]
        score_ncs = [i for sl in score_ncs for i in sl]

        fig, ax = plt.subplots()

        ax.hist(
            train_ncs,
            bins=nbins,
            stacked=True,
            density=True,
            alpha=0.5,
            color=colours[0],
        )
        ax.hist(
            update_ncs,
            bins=nbins,
            stacked=True,
            density=True,
            alpha=0.5,
            color=colours[1],
        )
        ax.hist(
            score_ncs,
            bins=nbins,
            stacked=True,
            density=True,
            alpha=0.5,
            color=colours[2],
        )
#         fig.legend(["train", "update", "score"])
        lgd = ax.legend(['train', 'update', 'score'], loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title(f"{endpoint}: distribution of nonconformity scores class {cl}")
        return plt, lgd

    @property
    def averaged_evaluation_df_pred_score(self):
        return self._average_evaluation_df(
            self._evaluation_df_pred_score, self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_pred_test(self):
        return self._average_evaluation_df(
            self._evaluation_df_pred_test, self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_cal_update(self):
        return self._average_evaluation_df(
            self._evaluation_df_cal_update, self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_cal_update2(self):
        return self._average_evaluation_df(
            self._evaluation_df_cal_update2, self.num_actives, self.num_inactives
        )

    @property
    def pred_predictions_df(self):
        return self._format_predictions_df(self._pred_predictions, self._names_predict)

    @property
    def cal_update_predictions_df(self):
        return self._format_predictions_df(
            self._cal_update_predictions, self._names_predict
        )


class CPTox21TrainUpdateCrossValidator(PredictCrossValidator):
    """
    An aggregated conformal predictor specifically adapted for the CPTox21 trainupdate part
    """

    def __init__(self, predictor, cv_splitter):
        super().__init__(predictor, cv_splitter)
        self._evaluation_df_pred_score = None
        self._pred_score_predictions = None

    def cross_validate(
        self,
        steps,
        endpoint=None,
        X_train=None,
        y_train=None,
        X_update=None,
        y_update=None,
        X_score=None,
        y_score=None,
        class_wise_evaluation=False,
    ):

        cv_predictions = []
        cv_y_test = []
        cv_predictors = []

        pred_score_predictions = []

        cv_evaluations = self._create_empty_evaluations_dict()
        pred_score_evaluations = self._create_empty_evaluations_dict()

        samples = self.sampler.gen_samples(labels=y_train)

        for fold, tr_idx, test_idx in samples:
            cv_y_test.append(y_train[test_idx])
            X_tr_update, y_tr_update = combine_csr(
                X_train[tr_idx], y_train[tr_idx], X_update, y_update
            )

            predictor = copy.deepcopy(self.predictor)

            # Fit ACP
            predictor.fit_calibrate(X_train=X_tr_update, y_train=y_tr_update)

            cv_predictors.append(predictor)
            cv_prediction = predictor.predict(X_score=X_train[test_idx])
            cv_predictions.append(cv_prediction)
            pred_score_prediction = predictor.predict(X_score=X_score)
            pred_score_predictions.append(pred_score_prediction)

            # Evaluate predictions
            for prediction, y_true, evaluations in zip(
                [cv_prediction, pred_score_prediction],
                [y_train[test_idx], y_score],
                [cv_evaluations, pred_score_evaluations],
            ):
                # fixme: do we really need evaluations???
                evaluations = self._evaluate(
                    prediction,
                    y_true,
                    evaluations,
                    endpoint=endpoint,
                    fold=fold,
                    steps=steps,
                    class_wise=class_wise_evaluation,
                )

        self._evaluation_df_cv = pd.DataFrame(cv_evaluations)
        self._evaluation_df_pred_score = pd.DataFrame(pred_score_evaluations)

        self._cv_predictions = [cv_predictions, cv_y_test]
        self._pred_score_predictions = [pred_score_predictions, y_score]

        self.cv_predictors = cv_predictors

        return pd.DataFrame(cv_evaluations), pd.DataFrame(pred_score_evaluations)

    def calibration_plot(
        self,
        averaged_evaluation_df,
        endpoint=None,
        colours=("blue", "darkred", "deepskyblue", "lightcoral"),
        class_wise=True,
        efficiency=True,
            title_name=None,
        **kwargs,
    ):

        return self._calibration_plot(
            averaged_evaluation_df=averaged_evaluation_df,
            endpoint=endpoint,
            colours=colours,
            class_wise=class_wise,
            efficiency=efficiency,
            title_name=title_name
        )

    @property
    def averaged_evaluation_df_pred_score(self):
        return self._average_evaluation_df(
            self._evaluation_df_pred_score, self.num_actives, self.num_inactives
        )


# --------------------------------
# Evaluator
# --------------------------------


class Evaluator:
    def __init__(self, y_pred, y_true=None, score_set=None, endpoint=None):
        if y_true is None:
            y_true = score_set.measurements[endpoint]
        #         print(y_pred[:5], type(y_pred))
        y_pred_0 = y_pred[:, 0]
        y_pred_1 = y_pred[:, 1]
        _prediction_df = pd.DataFrame(
            data={"p0": y_pred_0, "p1": y_pred_1, "known_label": y_true}
        )
        #         print(_prediction_df.shape)
        _prediction_df = (
            _prediction_df.dropna()
        )  # fixme: is this necessary? We only consider values == 0.0 and
        # values == 1.0 anyways
        #         print('after dropna: ', _prediction_df.shape)
        self._prediction_df = _prediction_df
        self.endpoint = endpoint

    def _calculate_set_sizes(self):
        nof_neg = float(sum(self._prediction_df["known_label"].values == 0.0))
        nof_pos = float(sum(self._prediction_df["known_label"].values == 1.0))
        nof_all = float(nof_neg + nof_pos)

        return nof_all, nof_neg, nof_pos

    def _calculate_nof_one_class_predictions(self, label, significance):
        """
        Calculate number of one class predictions for a specific class at a given significance level
        """

        # Get number of compounds that have respective label
        # and only one of the p-values fullfills significance level
        nof = sum(
            (self._prediction_df["known_label"].values == label)
            & (
                (
                    (self._prediction_df.p0.values < significance)
                    & (self._prediction_df.p1.values >= significance)
                )
                | (
                    (self._prediction_df.p0.values >= significance)
                    & (self._prediction_df.p1.values < significance)
                )
            )
        )

        return nof

    def calculate_efficiency(self, significance):
        """
           Calculate ratio of efficient predictions, i.e. prediction sets containig one single label
           """

        # Calculate total number of compounds, class-wise and all compounds
        total, total_0, total_1 = self._calculate_set_sizes()

        # Calculate number of efficiently predicted compounds
        # (only one label not in prediction set at given significance level)
        # class-wise
        efficiency_0 = self._calculate_nof_one_class_predictions(0.0, significance)
        efficiency_1 = self._calculate_nof_one_class_predictions(1.0, significance)

        # Calculate efficiency rate, class-wise and for all compounds
        efficiency_rate_0 = round(efficiency_0 / total_0, 3)
        efficiency_rate_1 = round(efficiency_1 / total_1, 3)
        efficiency_rate = round(((efficiency_0 + efficiency_1) / total), 3)

        return {
            "efficiency": efficiency_rate,
            "efficiency_0": efficiency_rate_0,
            "efficiency_1": efficiency_rate_1,
        }

    def calculate_validity(self, significance):
        """
           Calculate ratio of valid predictions, i.e. prediction sets containing the correct label
           """

        # Calculate total number of compounds, class-wise and all compounds
        total, total_0, total_1 = self._calculate_set_sizes()

        # Calculate number of wrongly predicted compounds
        # (correct label not in prediction set at given significance level)
        # class-wise
        error_0 = sum(
            (self._prediction_df["known_label"].values == 0.0)
            & (self._prediction_df.p0.values < significance)
        )
        error_1 = sum(
            (self._prediction_df["known_label"].values == 1.0)
            & (self._prediction_df.p1.values < significance)
        )

        # Calculate error rate, class-wise and for all compounds
        error_rate_0 = round(error_0 / total_0, 3)
        error_rate_1 = round(error_1 / total_1, 3)
        error_rate = round(((error_0 + error_1) / total), 3)

        return {
            "validity": (1 - error_rate),
            "validity_0": (1 - error_rate_0),
            "validity_1": (1 - error_rate_1),
        }

    def calculate_error_rate(self, significance):
        """
           Calculate ratio of valid predictions, i.e. prediction sets containing the correct label
           """

        # Calculate total number of compounds, class-wise and all compounds
        total, total_0, total_1 = self._calculate_set_sizes()

        # Calculate number of wrongly predicted compounds
        # (correct label not in prediction set at given significance level)
        # class-wise
        error_0 = sum(
            (self._prediction_df["known_label"].values == 0.0)
            & (self._prediction_df.p0.values < significance)
        )
        error_1 = sum(
            (self._prediction_df["known_label"].values == 1.0)
            & (self._prediction_df.p1.values < significance)
        )

        # Calculate error rate, class-wise and for all compounds
        error_rate_0 = round(error_0 / total_0, 3)
        error_rate_1 = round(error_1 / total_1, 3)
        error_rate = round(((error_0 + error_1) / total), 3)

        return {
            "error_rate": error_rate,
            "error_rate_0": error_rate_0,
            "error_rate_1": error_rate_1,
        }

    def calculate_accuracy(self, significance):
        """
          Calculate ratio of accurate predictions, i.e. efficient prediction sets containing the one correct label
          """
        # Calculate number of efficiently predicted compounds
        # (only one label not in prediction set at given significance level)
        # class-wise
        efficiency_0 = self._calculate_nof_one_class_predictions(0.0, significance)
        efficiency_1 = self._calculate_nof_one_class_predictions(1.0, significance)
        efficiency = efficiency_0 + efficiency_1

        # Calculate number of correctly and efficiently predicted compounds
        # (only one correct label in prediction set at given significance level)
        # class-wise
        accuracy_0 = sum(
            (self._prediction_df["known_label"].values == 0.0)
            & (self._prediction_df.p0.values >= significance)
            & (self._prediction_df.p1.values < significance)
        )
        accuracy_1 = sum(
            (self._prediction_df["known_label"].values == 1.0)
            & (self._prediction_df.p0.values < significance)
            & (self._prediction_df.p1.values >= significance)
        )

        # Calculate accuracy rate, class-wise and for all compounds
        # todo: how to handle division by zero??
        accuracy_rate_0 = (
            round(accuracy_0 / efficiency_0, 3) if efficiency_0 != 0 else 0
        )
        accuracy_rate_1 = (
            round(accuracy_1 / efficiency_1, 3) if efficiency_1 != 0 else 0
        )
        accuracy_rate = (
            round(((accuracy_0 + accuracy_1) / efficiency), 3) if efficiency != 0 else 0
        )

        return {
            "accuracy": accuracy_rate,
            "accuracy_0": accuracy_rate_0,
            "accuracy_1": accuracy_rate_1,
        }

    def calibration_plot(self, steps):
        # fixme: I am not sure yet, if this method should live here
        # todo: include class-wise evaluation
        validities_tot = [
            self.calculate_validity(i / float(steps))["validity"] for i in range(steps)
        ] + [self.calculate_validity(1)["validity"]]
        error_rate_tot = [1 - i for i in validities_tot]
        efficiencies_tot = [
            self.calculate_efficiency(i / float(steps))["efficiency"]
            for i in range(steps)
        ] + [self.calculate_efficiency(1)["efficiency"]]
        sl = [i / float(steps) for i in range(steps)] + [1]
        print(sl)

        fig, ax = plt.subplots()

        ax.plot([0, 1], [0, 1], "--", linewidth=1, color="black")
        ax.plot(sl, error_rate_tot, label=self.endpoint)
        ax.plot(sl, efficiencies_tot)
        ax.legend(loc="lower right")
        ax.xlabel("significance")
        ax.ylabel("error rate")
        return fig

    # ---------------
    # RMSD evaluation
    # ---------------
    
    
def calculate_deviation_square(error, sl):
    """
    Calculate the square deviation between a given error value and a significance level
    Parameters
    ----------
    error : error
    sl : significance level

    Returns
    -------
    square deviation

    """
    return (error-sl)**2

    
def calculate_rmsd_from_df(eval_df, cl=None):
    """
    Calculate the rmsd (root mean square deviation) for all error-significance level pairs
    in a dataframe
    
    Parameters
    ----------
    eval_df : dataframe for which the rmsd should be calculated
    cl : class of compounds for which the rmsd should be calculated, i.e. 0 or 1
    if cl is None, the overall rmsd for all compounds will be calculated.
    
    Returns
    -------
    dataframe with an additional 'rmsd' column
    
    """
    if cl:
        eval_df['square'] = eval_df.apply(lambda row: calculate_deviation_square(
            row[f"error_rate_{cl} mean"],   row["significance_level"]), axis=1)
    else:
        eval_df['square'] = eval_df.apply(lambda row: calculate_deviation_square(
            row["error_rate mean"], row["significance_level"]), axis=1)
    rmsd = np.round(math.sqrt(np.mean(eval_df["square"])), 3)
    
    return rmsd

def calculate_pos_deviation_square(error, sl):
    """
    Calculate the square deviation between a given error value and a significance level
    if the deviation is positive (>0)
    
    Parameters
    ----------
    error : error
    sl : significance level
    
    Returns
    -------
    square deviation or 0
    
    """
    
    if error > sl:
        return (error-sl)**2
    else:
        return 0


def calculate_rmsd_pos_from_df(eval_df, cl=None):
    # fixme: exchange 'rmsd_pos' with a more appropriate term
    """
    Calculate the rmsd (root mean square deviation) for all error-significance level pairs
    in a dataframe if the deviation (error - significance level) is larger than 0
    
    Parameters
    ----------
    eval_df : dataframe for which the rmsd_pos should be calculated
    cl : class of compounds for which the rmsd_pos should be calculated, i.e. 0 or 1
        if cl is None, the overall rmsd_pos for all compounds will be calculated.
    
    Returns
    -------
    dataframe with an additional 'rmsd_pos' column
    
    """
    if cl:
        eval_df['square'] = eval_df.apply(lambda row: calculate_pos_deviation_square(
            row[f"error_rate_{cl} mean"],   row["significance_level"]), axis=1)
    else:
        eval_df['square'] = eval_df.apply(lambda row: calculate_pos_deviation_square(
            row["error_rate mean"], row["significance_level"]), axis=1)
    rmsd_pos = np.round(math.sqrt(np.mean(eval_df["square"])), 3)
    
    return rmsd_pos
