from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sdk():
    client = MagicMock()
    database = client.get_database_client.return_value
    database.list_containers.return_value = [{"id": "orders"}, {"id": "private"}]
    container = database.get_container_client.return_value
    container.read.return_value = {
        "id": "orders",
        "partitionKey": {"paths": ["/tenant"]},
        "indexingPolicy": {"indexingMode": "consistent"},
    }
    container.query_items.return_value = iter([{"id": "1"}])
    return client, database, container
