# This script is part of the supporting information to the manuscript entitled "Conformal Prediction and
# Exchangeability in Toxicological In Vitro Datasets (title tbd)". The script was developed by Andrea Morger in the
# In Silico Toxicology and Structural Biology Group of Prof. Dr. Andrea Volkamer at the Charité Universitätsmedizin
# Berlin, in collaboration with Fredrik Svensson, Ulf Norinder and Ola Spjuth. It was last updated in September 2020.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cptox21 import (
    define_path,
    load_signatures_files,
    CPTox21CrossValidator,
    CrossValidationSampler,
    StratifiedRatioSampler,
    CPTox21TrainUpdateCrossValidator,
)


def load_data(endpoint, signatures_path, short_train=False):
    """
    Load signature datasets per endpoint
    
    Parameters
    ----------
    signatures_path : path to signatures, which should be loaded
    short_train : set to true, if only small part of training set should be used, e.g. for test run
    endpoint : endpoint for which the data should be loaded
    
    Returns
    -------
    X_train : (signature) descriptors for Tox21train set
    y_train : labels for Tox21train set
    X_test : (signature) descriptors for Tox21test set
    y_test : labels for Tox21test set
    X_score : (signature) descriptors for Tox21score set
    y_score : labels for Tox21score set
    
    """
    dataset_names = ["train", "test", "score"]
    train_path = os.path.join(
        signatures_path, f"data_signatures_{endpoint}_{dataset_names[0]}.csr"
    )
    test_path = os.path.join(
        signatures_path, f"data_signatures_{endpoint}_{dataset_names[1]}.csr"
    )
    score_path = os.path.join(
        signatures_path, f"data_signatures_{endpoint}_{dataset_names[2]}.csr"
    )

    X_train, y_train, X_test, y_test, X_score, y_score = load_signatures_files(
        train_path, test_path, score_path
    )

    if short_train:
        X_train = X_train[:500]
        y_train = y_train[:500]

    return X_train, y_train, X_test, y_test, X_score, y_score


def cross_validate_compare_calibration_sets(
    endpoint, acp, X_train, y_train, X_test, y_test, X_score, y_score, n_cv=5, random_state=None
):
    """
    Perform a crossvalidation, including the following settings:
    * use original training and calibration set (Tox21train): predict internal test set (Tox21train)
    * use original training and calibration set: predict score set (Tox21score)
    * use original training and calibration set: predict test set (Tox21test)
    * update calibration set with Tox21test: predict score set
    * update calibration set with part of Tox21score: predict (other) part of Tox21score
    
    Parameters
    ----------
    random_state : define a random state for reproducibility
    n_cv : number of folds in cross-validation
    endpoint : endpoint for which the data should be loaded
    
    X_train : (signature) descriptors for Tox21train set
    y_train : labels for Tox21train set
    X_test : (signature) descriptors for Tox21test set
    y_test : labels for Tox21test set
    X_score : (signature) descriptors for Tox21score set
    y_score : labels for Tox21score set
    
    Returns
    -------
    cross_validator : cross_validator class with fitted and calibrated models and evaluation dfs
    """
    cross_validator = CPTox21CrossValidator(
        acp,
        cv_splitter=CrossValidationSampler(n_cv, random_state=random_state),
        score_splitter=StratifiedRatioSampler(test_ratio=0.5, random_state=random_state),
    )
    cross_validation_dfs = cross_validator.cross_validate(
        steps=10,
        endpoint=endpoint,
        X_train=X_train,
        y_train=y_train,
        X_update=X_test,
        y_update=y_test,
        X_score=X_score,
        y_score=y_score,
    )
    return cross_validator


def cross_validate_with_updated_training_set(
    endpoint,
    train_update_acp,
    X_train,
    y_train,
    X_test,
    y_test,
    X_score,
    y_score,
    known_indices_sampler,
):

    """
    Perform a crossvalidation, including the following settings:
    * update original training set (Tox21train) with Tox21test: predict internal test set (Tox21train & Tox21test)
    * update original training set (Tox21train) with Tox21test: predict score set (Tox21score)
        
    Parameters
    ----------
    train_update_acp : acp used to train model with updated training set
    endpoint : endpoint for which the data should be loaded
    
    X_train : (signature) descriptors for Tox21train set
    y_train : labels for Tox21train set
    X_test : (signature) descriptors for Tox21test set
    y_test : labels for Tox21test set
    X_score : (signature) descriptors for Tox21score set
    y_score : labels for Tox21score set
    
    known_indices_sampler: Sampler to split X_train and y_train in the same train and test sets
    as used (known) for the previous experiments
    
    Returns
    -------
    cross_validator : cross_validator class with fitted and calibrated models and evaluation dfs
    """
    train_update_cross_validator = CPTox21TrainUpdateCrossValidator(
        train_update_acp, cv_splitter=known_indices_sampler
    )

    train_update_cross_validation_dfs = train_update_cross_validator.cross_validate(
        steps=10,
        endpoint=endpoint,
        X_train=X_train,
        y_train=y_train,
        X_update=X_test,
        y_update=y_test,
        X_score=X_score,
        y_score=y_score,
        class_wise_evaluation=False,
    )
    return train_update_cross_validator


def boxplot_rmsd(rmsds, rmsd_title, strategies=None):
    if strategies is None:
        strategies = [
            "cv_original",
            "pred_score_original",
            "pred_score_trainupdate",
            "pred_score_calupdate",
            "pred_score_calupdate2",
        ]
    """
    Generate a boxplot with the rmsd values over multiple endpoints.
    This function can be used to plot both rmsd or rmsd_pos values.
    
    Parameters
    ----------
    strategies : strategies or set-ups used when making the predictions (e.g. "original_cv")
    rmsds : a dictionary with the strategies as keys and a list of rmsd values for all the
        endpoints as values
    rmsd_title : the naming for 'rmsd' which should be used in the plot title, e.g. "rmsd", "rmsd_pos"
    """
    plt.clf()
    plt.boxplot([rmsds[k] for k in strategies], labels=strategies)
    plt.xticks(rotation="vertical")
    plt.title(f"{rmsd_title} over all endpoints")


def draw_calibration_plot_all_endpoints(
    endpoints,
    strategy,
    path,
    colours=("blue", "darkred", "deepskyblue", "lightcoral"),
    class_wise=True,
    efficiency=True,
    title_name=None,
):

    plt.clf()
    fig, axs = plt.subplots(ncols=4, nrows=3)
    fig.set_figheight(15)
    fig.set_figwidth(20)

    xax = 0
    yax = 0

    if class_wise and efficiency:
        evaluation_measures = [
            "error_rate_0",
            "error_rate_1",
            "efficiency_0",
            "efficiency_1",
        ]

    for endpoint in endpoints:
        eval_df = pd.read_csv(os.path.join(path, f"{endpoint}_averaged_eval_df_{strategy}.csv"))
        axs[xax, yax].plot([0, 1], [0, 1], "--", linewidth=1, color="black")
        sl = eval_df["significance_level"]
        for ev, colour in zip(evaluation_measures, colours):

            ev_mean = eval_df[f"{ev} mean"]
            ev_std = eval_df[f"{ev} std"]
            axs[xax, yax].plot(sl, ev_mean, label=True, c=colour)
            axs[xax, yax].fill_between(
                sl, ev_mean - ev_std, ev_mean + ev_std, alpha=0.3, color=colour
            )

        major_ticks = np.arange(0, 101, 20)
        minor_ticks = np.arange(0, 101, 5)

        axs[xax, yax].set_xticks(minor_ticks / 100.0, minor=True)
        axs[xax, yax].set_yticks(major_ticks / 100.0)
        axs[xax, yax].set_yticks(minor_ticks / 100.0, minor=True)

        axs[xax, yax].grid(which="minor", linewidth=0.5)  # alpha=0.5)
        axs[xax, yax].grid(which="major", linewidth=1.5)  # alpha=0.9, linewidth=2.0)

        axs[xax, yax].set_title(endpoint, fontsize=16)
        axs[xax, yax].set_xlabel("significance")
        axs[xax, yax].set_ylabel("error rate")
        par1 = axs[xax, yax].twinx()
        par1.set_ylabel("efficiency (SCP)")

        xax += 1
        if xax == 3:
            xax = 0
            yax += 1

        eval_legend = evaluation_measures.copy()
        eval_legend.insert(0, "expected_error_rate")

    lgd = fig.legend(eval_legend, loc="center left", bbox_to_anchor=(1, 0.47))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(strategy, fontsize=20)

    return plt, lgd
