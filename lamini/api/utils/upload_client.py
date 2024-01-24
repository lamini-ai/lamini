import hashlib
import json
import os
from azure.storage.blob import BlobClient, BlobBlock
import uuid
from tqdm import tqdm
import jsonlines


def upload_to_blob(data: str, sas_url: str):
    '''
    Upload large data to blob
    '''

    blob_client_sas = BlobClient.from_blob_url(blob_url=sas_url)
    if blob_client_sas.exists():
        print(f"File/data already exists")

    else:
        try:
            # upload in chunks
            block_list = []
            chunk_size = 5000
            for i in tqdm(range(0, len(data), chunk_size)):
                read_data = data[i:i + chunk_size]
                if not read_data:
                    break  # done
                blk_id = str(uuid.uuid4())
                data_str = '\n'.join([json.dumps(x) for x in read_data])
                blob_client_sas.stage_block(block_id=blk_id, data=data_str+"\n")
                block_list.append(BlobBlock(block_id=blk_id))
            blob_client_sas.commit_block_list(block_list)
        except BaseException as e:
            print(f'Upload file error :{e}')
            raise e


# async def upload_to_blob(data: str, sas_url: str):
#     blob_client_sas = BlobClient.from_blob_url(blob_url=sas_url)
#
#     async with blob_client_sas as blob:
#         if await blob.exists():
#             print(f"File/data already exists")
#
#         else:
#             print("Uploading data....")
#             s = time.time()
#             data_str = json.dumps(data)
#             e = time.time()
#             print(f"json dump done {e-s}")
#             await blob.upload_blob(data_str, blob_type="AppendBlob")
#             e2 = time.time()
#             print(f"Upload to blob completed for data.{e2-e}")


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
