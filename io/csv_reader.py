from reader import AbstractReader
from pandas import DataFrame


class CVSReader(AbstractReader):
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
        pass

    def _clean_data(self, raw_data: DataFrame) -> DataFrame:
        """Apply cleaning steps to raw, ingested data

        Arguments:
            raw_data: the raw data ingested from the data source
        """
        pass

    def _transform_data(self, cleaned_data: DataFrame):
        """Apply any processing steps needed to get the data in a `DataFrame` of
        cleaned, ingested data into the correct units, type, fp accuracy etc.

        Arguments:
            cleaned_data: the raw ingested data in a DataFrame
        """
        pass
