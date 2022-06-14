import os
import re

from astropy.io.votable import parse_single_table
import numpy as np

from sccala.asynphot import __path__ as ASYNPHOT_PATH


def load_transmission_data(filter_id):
    """
    Loads transmission data for requested filter_id from filter directory

    :param filter_id: str
        Filter ID in either 'facility/instrument/filter' or
        'facility/instrument.filter' format (can use'/' and '.' interchangeably as delimiters

    :return:
    """

    facility, instrument, filter_name = re.split("/|\.", filter_id)
    transmission_data_loc = os.path.join(
        ASYNPHOT_PATH[0],
        "filters",
        facility,
        instrument,
        "{:s}.vot".format(filter_name),
    )

    if not os.path.exists(transmission_data_loc):
        raise FileNotFoundError("Requested filter not found...")

    return df_from_votable(transmission_data_loc)


def df_from_votable(votable_path):
    table = parse_single_table(votable_path).to_table()
    df = table.to_pandas()
    return byte_to_literal_strings(df)


def byte_to_literal_strings(dataframe):
    """Converts byte strings (if any) present in passed dataframe to literal
    strings and returns an improved dataframe.
    """
    # Select the str columns:
    str_df = dataframe.select_dtypes([np.object])

    if not str_df.empty:
        # Convert all of them into unicode strings
        str_df = str_df.stack().str.decode("utf-8").unstack()
        # Swap out converted cols with the original df cols
        for col in str_df:
            dataframe[col] = str_df[col]

    return dataframe
