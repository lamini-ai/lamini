import itertools
import os
from typing import Iterable, Generator, Any

import jsonlines
from azure.storage.blob import BlobClient


def upload_to_blob(data: Iterable[str], sas_url: str) -> None:
    """Upload the provided data to the sas_url

    Parameters
    ----------
    data: Iterable[str]
        Data to upload

    sas_url: str
        Location to upload to

    Returns
    -------
    None
    """

    blob_client_sas = BlobClient.from_blob_url(blob_url=sas_url)

    if blob_client_sas.exists():
        print(f"\nFile/data already exists")

    else:
        print("\nUploading data....")
        blob_client_sas.upload_blob(data)
        print(f"Upload to blob completed for data.")


def upload_to_local(data: Iterable[str], dataset_location: str) -> None:
    """Upload provided data to local storage

    Parameters
    ----------
    data: Iterable[str]
        Data to upload

    dataset_location: str
        Local location to store data

    Returns
    -------
    None
    """

    if os.path.exists(dataset_location):
        print(f"File/data already exists")
    else:
        print("Uploading data....")
        with jsonlines.open(dataset_location, "w") as f:
            for row in data:
                f.write(row)
        print(f"Upload completed for data.")


class SerializableGenerator(list):
    """Generator that is serializable by JSON to send uploaded data over http requests"""

    def __init__(self, iterable) -> None:
        tmp_body = iter(iterable)
        try:
            self._head = iter([next(tmp_body)])
            self.append(tmp_body)
        except StopIteration:
            self._head = []

    def __iter__(self) -> Generator[Any, None, None]:
        return itertools.chain(self._head, *self[:1])
