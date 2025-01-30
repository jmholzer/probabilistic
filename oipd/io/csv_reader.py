import numpy as np
import pandas as pd
from pandas import DataFrame

from oipd.io.reader import AbstractReader


class CSVReader(AbstractReader):
    """Reader implementation for pre-formatted CSV files containing options data

    Methods:
        _ingest_data
        _clean_data
        _transform_data
    """

    def _ingest_data(self, url: str) -> DataFrame:
        """Ingest raw data from a data source at the given URL

        Arguments:
            url: the url to retrieve the raw data from
        """
        return pd.read_csv(url)

    def _clean_data(self, raw_data: DataFrame) -> DataFrame:
        """Apply cleaning steps to raw, ingested data

        Arguments:
            raw_data: the raw data ingested from the data source
        """
        raw_data["strike"] = raw_data["strike"].astype(np.float64)
        raw_data["last_price"] = raw_data["last_price"].astype(np.float64)
        raw_data["bid"] = raw_data["bid"].astype(np.float64)
        raw_data["ask"] = raw_data["ask"].astype(np.float64)
        return raw_data

    def _transform_data(self, cleaned_data: DataFrame):
        """Apply any processing steps needed to get the data in a `DataFrame` of
        cleaned, ingested data into the correct units, type, fp accuracy etc.

        Arguments:
            cleaned_data: the raw ingested data in a DataFrame
        """

        return cleaned_data
