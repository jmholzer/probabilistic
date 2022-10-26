from abc import ABC, abstractmethod
from pandas import DataFrame


class AbstractReader(ABC):
    """Abstract class for readers -- ingest data from a source
    and return the cleaned, transformed result as a DataFrame

    Methods:
        __init__
        read
        _ingest_data
        _clean_column_names
    """

    @abstractmethod
    def read(self, url: str) -> DataFrame:
        """The main API endpoint for Reader objects. Read the data at a given
        URL into a DataFrame, clean its column names and return it.

        Arguments:
            url: the url to retrieve the raw data from
        """
        if not url:
            raise ValueError("url is empty or invalid")
        raw_data = self._ingest_data(url)
        cleaned_data = self._clean_data(raw_data)
        transformed_cleaned_data = self._transform_data(cleaned_data)
        return transformed_cleaned_data

    @abstractmethod
    def _ingest_data(self, url: str) -> DataFrame:
        """Ingest raw data from a data source at the given URL

        Arguments:
            url: the url to retrieve the raw data from
        """
        raise NotImplementedError("`_ingest_data` method has not been implemented")

    @abstractmethod
    def _clean_data(self, raw_data: DataFrame) -> DataFrame:
        """Apply cleaning steps to raw, ingested data

        Arguments:
            raw_data: the raw data ingested from the data source
        """
        raise NotImplementedError("`_clean_data` method has not been implemented")

    @abstractmethod
    def _transform_data(self, cleaned_data: DataFrame):
        """Apply any processing steps needed to get the data in a `DataFrame` of
        cleaned, ingested data into the correct units, type, fp accuracy etc.

        Arguments:
            cleaned_data: the raw ingested data in a DataFrame
        """
        raise NotImplementedError("_transform_data method has not been implemented")
