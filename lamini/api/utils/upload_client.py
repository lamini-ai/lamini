import hashlib
import os
import time
from typing import Dict, Iterable, List, Union

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
            f.write_all(data)
        print(f"Upload completed for data.")


def get_dataset_name():
    m = hashlib.sha256()
    m.update(str(time.time()).encode("utf-8"))
    return m.hexdigest()
