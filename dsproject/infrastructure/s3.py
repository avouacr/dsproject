from datetime import date

import pyarrow.parquet as pq
import s3fs
import pyarrow as pa


class S3Client:
    """
    A class to represent a S3 client for input output.
    ...
    Attributes
    ----------
    fs : s3fs.S3FileSystem
        A s3 file system client
    Methods
    -------
    write_df_to_s3(df, bucket_name, path):
         Write a pandas data frame to s3 in parquet format
    read_parquet_from_s3(self, bucket_name, path):
         Read a parquet file from s3 and return it as a pandas data frame
    """
    def __init__(self, endpoint, access_key, access_secret, token):
        """Constructs an instance of s3fs client
         Parameters
         ----------
             endpoint : str
                 endpoint of your s3 server
             access_key : str
                 access_key of your s3 account
             access_secret : str
                 access_secret of your s3 account
             token: str
                 access_token of your s3 account
        """
        url = f"https://{endpoint}"
        self.fs = s3fs.S3FileSystem(key=access_key, secret=access_secret, token=token,
                                    client_kwargs={'endpoint_url': url})

    # This function write a pandas dataframe to s3 in parquet format
    def write_df_to_s3(self, df, bucket_name, path):
        """ Write pandas data frame to s3
        Parameters
        ----------
             df : pandas.DataFrame
                 a pandas data frame that stores the tweet message
             bucket_name : str
                 the name of your s3 bucket
             path : str
                 the path that you want to store your data
        Returns
        -------
              None
          """
        # Convert pandas df to Arrow table
        table = pa.Table.from_pandas(df)
        file_uri = f"{bucket_name}/{path}"
        pq.write_to_dataset(table, root_path=file_uri, filesystem=self.fs)

    # This function read a parquet file and return a arrow table
    def read_parquet_from_s3(self, bucket_name, path):
        """ Read a parquet file from s3
        Parameters
        ----------
             bucket_name : str
                 the name of your s3 bucket
             path : str
                 the path that you want to store your data
        Returns
        -------
             df : pandas.DataFrame
                a pandas data frame that stores the tweet message
          """
        file_uri = f"{bucket_name}/{path}"
        dataset = pq.ParquetDataset(file_uri, filesystem=self.fs)
        return dataset.read().to_pandas()
