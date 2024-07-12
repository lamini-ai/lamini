import unittest
from unittest.mock import patch

import jsonlines
import lamini
from lamini.api.lamini import Lamini
from lamini.api.train import Train


def mock_get_dataset_location(upload_base_path, dataset_id, is_public, data=None):
    return {"dataset_location": "/tmp/read"}


def mock_train_api(key, url, http_method, json=None):
    return {"job_id": 1}


def mock_blob_creation(key, url, http_method, json=None):
    return {"dataset_location": "location"}


class MockBlobClient:
    def from_blob_url(self, blob_url):
        assert blob_url is not None
        return self

    def exists(self):
        return self

    def upload_blob(self, data):
        assert data == ["test_data"]
        return self


def mock_from_blob_url(blob_url):
    assert blob_url is not None
    return MockBlobClient()


class TestTrainDataUpload(unittest.TestCase):
    def setUp(self):
        self.trainer = Train(api_key="test_k", api_url="test_url")

    @patch("lamini.api.train.make_web_request")
    def test_train_api_payload(self, mock_make_web_request):
        mock_make_web_request.side_effect = mock_train_api

        job = self.trainer.train(
            model_name="hf-internal-testing/tiny-random-gpt2",
            dataset_id="test_dataset",
        )
        assert job["job_id"] == 1

    @patch("lamini.api.train.make_web_request")
    def test_create_blob_dataset_location(self, mock_make_web_request):
        mock_make_web_request.side_effect = mock_blob_creation

        location = self.trainer.create_blob_dataset_location(
            "test_path",
            "test_dataset",
            True,
            [{"input": "test_data", "output": "test_data"}],
        )
        assert location["dataset_location"] == "location"

    @patch("lamini.api.lamini.get_dataset_name")
    @patch("lamini.api.train.Train.update_blob_dataset_num_datapoints")
    @patch("lamini.api.train.Train.create_blob_dataset_location")
    @patch("lamini.api.utils.upload_client.BlobClient.from_blob_url")
    @patch("lamini.api.train.Train.get_upload_base_path")
    def test_upload_data(
        self,
        mock_get_upload_base_path,
        mock_block_client,
        mock_create_blob_dataset_location,
        mock_update_blob_dataset_num_datapoints,
        mock_get_dataset_name,
    ):
        mock_block_client.side_effect = mock_from_blob_url
        mock_get_upload_base_path.return_value = {"upload_base_path": "azure"}
        mock_create_blob_dataset_location.side_effect = mock_get_dataset_location
        mock_update_blob_dataset_num_datapoints.return_value = {"status": "success"}
        mock_get_dataset_name.return_value = "test_dataset_id"
        lamini = Lamini(
            api_key="test_k",
            api_url="test",
            model_name="hf-internal-testing/tiny-random-gpt2",
        )
        dataset_id = lamini.upload_data(["test_data"], True)
        assert dataset_id == "test_dataset_id"
