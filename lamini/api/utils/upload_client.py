from azure.storage.blob import BlobClient

import json
import hashlib
import os
import jsonlines


def upload_to_blob(data: str, sas_url: str):
    blob_client_sas = BlobClient.from_blob_url(blob_url=sas_url)

    if blob_client_sas.exists():
        print(f"File/data already exists")

    else:
        print("Uploading data....")
        data_str = json.dumps(data)
        blob_client_sas.upload_blob(data_str, blob_type="AppendBlob")
        print(f"Upload to blob completed for data.")


def upload_to_local(data, dataset_location):
    if os.path.exists(dataset_location):
        print(f"File/data already exists")
    else:
        print("Uploading data....")
        with jsonlines.open(dataset_location, "w") as f:
            f.write_all(data)
        print(f"Upload completed for data.")


def get_dataset_name(dataset):
    m = hashlib.sha256()
    m.update(str(dataset).encode("utf-8"))
    return m.hexdigest()
