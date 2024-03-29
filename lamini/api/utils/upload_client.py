import hashlib
import os
import time
from typing import Dict, Iterable, List, Union
import itertools
import jsonlines
from azure.storage.blob import BlobClient


def upload_to_blob(data: Iterable[str], sas_url: str):
    blob_client_sas = BlobClient.from_blob_url(blob_url=sas_url)

    if blob_client_sas.exists():
        print(f"\nFile/data already exists")

    else:
        print("\nUploading data....")
        blob_client_sas.upload_blob(data)
        print(f"Upload to blob completed for data.")


def upload_to_local(data, dataset_location):
    if os.path.exists(dataset_location):
        print(f"File/data already exists")
    else:
        print("Uploading data....")
        with jsonlines.open(dataset_location, "w") as f:
            for row in data:
                f.write(row)
        print(f"Upload completed for data.")


def get_dataset_name():
    m = hashlib.sha256()
    m.update(str(time.time()).encode("utf-8"))
    return m.hexdigest()


class SerializableGenerator(list):
    """Generator that is serializable by JSON to send uploaded data over http requests"""

    def __init__(self, iterable):
        tmp_body = iter(iterable)
        try:
            self._head = iter([next(tmp_body)])
            self.append(tmp_body)
        except StopIteration:
            self._head = []

    def __iter__(self):
        return itertools.chain(self._head, *self[:1])
