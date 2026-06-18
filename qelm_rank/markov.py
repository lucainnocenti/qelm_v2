"""Not sure what this module is for. Probably obsolete stuff"""


import numpy as np
import pandas as pd


def add_markov_slack_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit a_p choices based on the two common B variables.

    For B -> 0, a_p = B^{-1/4}.
    Then failure Markov bound is a_p^{-2}=B^{1/2},
    and a_p^2 B = B^{1/2}.
    """
    df = df.copy()

    eps = 1e-300

    if "B_general_worst_cp1" in df.columns:
        B = np.maximum(df["B_general_worst_cp1"].astype(float), eps)
        df["a_p_worst_cp1"] = B ** (-0.25)
        df["markov_fail_bound_worst_cp1"] = B ** 0.5
        df["a_p_squared_B_worst_cp1"] = (df["a_p_worst_cp1"] ** 2) * B

    if "B_cp_qminus1" in df.columns:
        B = np.maximum(df["B_cp_qminus1"].astype(float), eps)
        df["a_p_cp_qminus1"] = B ** (-0.25)
        df["markov_fail_bound_cp_qminus1"] = B ** 0.5
        df["a_p_squared_B_cp_qminus1"] = (df["a_p_cp_qminus1"] ** 2) * B

    return df
