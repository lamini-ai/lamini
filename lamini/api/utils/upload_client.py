from azure.storage.blob.aio import BlobServiceClient, BlobClient

import hashlib
import os
import jsonlines
import asyncio
import json


async def async_upload_to_blob(data: str, sas_url: str):
    blob_client_sas = BlobClient.from_blob_url(blob_url=sas_url)
    async with blob_client_sas as blob:
        if await blob.exists():
            print(f"File/data already exists")

        else:
            print("Uploading data....")
            await blob.upload_blob(data, blob_type="AppendBlob")
            print(f"Upload to blob completed for data.")


def upload_to_blob(data: str, sas_url: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        tasks = [loop.create_task((async_upload_to_blob(json.dumps(data), sas_url)))]
        done, pending = loop.run_until_complete(asyncio.wait(tasks))
        results = []
        for future in done:
            url = future.result()
            results.append(url)
        return results[0]
    except Exception as e:
        print(f"\n\nError while running async upload task for file/data.")
        raise e
    finally:
        loop.close()


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
