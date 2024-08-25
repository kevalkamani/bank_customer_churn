"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from model.config.core import config
from model.processing.features import *


def test_columndrop_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    transformer = ColumnDropper(col_list=config.modl_config.cols_delete)

    assert (
        all([True for col in config.modl_config.cols_delete if col in df_test.columns])
        == True
    )

    # When
    subject = transformer.fit(df_test).transform(df_test)

    # Then
    assert (
        all(
            [
                True
                for col in config.modl_config.cols_delete
                if not col in subject.columns
            ]
        )
        == True
    )


def test_binner_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    df_temp = df_test.copy()
    for col in config.modl_config.label_cols:

        if "age" in col:
            a = config.modl_config.age_binner
            b = config.modl_config.age_bins
            c = config.modl_config.age_bin_labels
        elif "balance" in col:
            a = config.modl_config.balance_binner
            b = config.modl_config.bal_bins
            c = config.modl_config.bal_bin_labels
        elif "tenure" in col:
            a = config.modl_config.tenure_binner
            b = config.modl_config.ten_bins
            c = config.modl_config.ten_bin_labels

        transformer = Binner(a, b, c)

        subject = transformer.fit(df_test).transform(df_test)

        assert type(subject[col][0]) == str
        assert subject[col].nunique() == len(transformer.labels)
        assert all([True for lbl in transformer.labels if lbl in subject[col]]) == True


def test_mapper_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    transformer = Mapper(col_map=config.modl_config.mapping_dict)

    for key, val in config.modl_config.mapping_dict.items():

        assert all([True for i in list(val.keys()) if i in df_test[key]])

        # When
        subject = transformer.fit(df_test).transform(df_test)

        # Then
        assert all([True for i in list(val.values()) if i in subject[key]])


def test_outlier_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    transformer = OutlierHandler(col_list=config.modl_config.num_cols)

    for col in config.modl_config.num_cols:

        q1 = df_test.describe()[col].loc["25%"]
        q3 = df_test.describe()[col].loc["75%"]
        iqr = q3 - q1
        lower_bound = int(q1 - (1.5 * iqr))
        upper_bound = int(q3 + (1.5 * iqr))

        print(lower_bound)
        print(upper_bound)

        assert (
            len(df_test[df_test[col] > upper_bound])
            + len(df_test[df_test[col] < lower_bound])
            >= 0
        )

        # When
        subject = transformer.fit(df_test).transform(df_test)

        # Then
        assert (
            len(subject[subject[col] > upper_bound])
            + len(subject[subject[col] < lower_bound])
        ) == 0


def test_onehot_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    transformer = ColOneHotEncoder(col_list=config.modl_config.onehot_cols)

    for col in config.modl_config.onehot_cols:

        assert (
            all([True for cat in transformer.categories_ if cat in df_test[col]])
            == True
        )

        # When
        subject = transformer.fit(df_test).transform(df_test)

        # Then
        assert (
            sum(subject[[clmn for clmn in subject.columns if col in clmn]].iloc[0, :])
            == 1
        )
        assert len(
            subject[[clmn for clmn in subject.columns if col in clmn]].columns
        ) == len(transformer.categories_[col])


def test_label_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    df_temp = df_test.copy()
    for col in config.modl_config.label_cols:

        if "age" in col:
            a = config.modl_config.age_binner
            b = config.modl_config.age_bins
            c = config.modl_config.age_bin_labels
        elif "balance" in col:
            a = config.modl_config.balance_binner
            b = config.modl_config.bal_bins
            c = config.modl_config.bal_bin_labels
        elif "tenure" in col:
            a = config.modl_config.tenure_binner
            b = config.modl_config.ten_bins
            c = config.modl_config.ten_bin_labels

        transformer1 = Binner(a, b, c)

        subject1 = transformer1.fit(df_temp).transform(df_temp)
        df_temp = subject1.copy()

    transformer2 = ColLabelEncoder(col_list=config.modl_config.label_cols)
    subject2 = transformer2.fit(subject1).transform(subject1)
    # Then
    for col in config.modl_config.label_cols:

        assert subject1[col].nunique() == len(
            transformer2.encoders[col].transform(transformer2.encoders[col].classes_)
        )
        assert subject2[col].dtype == int
        assert (
            all([True for lbl in transformer2.encoders.keys() if lbl in subject2[col]])
            == True
        )
