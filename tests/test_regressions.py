import json
from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import ServiceRequestError, ServiceResponseError
from azure.core.paging import ItemPaged
from azure.cosmos.database import DatabaseProxy
from azure.cosmos.exceptions import CosmosClientTimeoutError, CosmosHttpResponseError
from langchain_core.language_models.fake import FakeListLLM
from langchain_google_genai._function_utils import (
    convert_to_genai_function_declarations,
)

from azure_cosmos_db_toolkit import (
    CosmosDBDatabase,
    CosmosDBDatabaseToolkit,
    InfoCosmosDBDatabaseTool,
    QueryCosmosDBDatabaseTool,
)
from azure_cosmos_db_toolkit.tool import QueryParameter


def test_explicit_container_names_need_no_listing_permission(sdk):
    client, database, _ = sdk
    database.list_containers.side_effect = CosmosHttpResponseError(status_code=403)
    for names in (["orders"], []):
        db = CosmosDBDatabase(client, "app", include_containers=names)
        assert db.get_usable_container_names() == names
        assert db.get_context() == {"container_names": names}
    database.list_containers.assert_not_called()


def test_unconfigured_containers_are_discovered(sdk):
    db = CosmosDBDatabase(sdk[0], "app")
    assert db.get_usable_container_names() == ["orders", "private"]


@pytest.mark.parametrize("value", ["abc", 42, 2.5, True, False, None])
def test_scalar_parameter_types_survive_tool_invocation(sdk, value):
    tool = QueryCosmosDBDatabaseTool(db=CosmosDBDatabase(sdk[0], "app"))
    tool.invoke(
        {
            "container": "orders",
            "query": "SELECT * FROM c WHERE c.value = @v",
            "parameters": [{"name": "@v", "value": value}],
            "partition_key": "tenant",
        }
    )
    actual = sdk[2].query_items.call_args.kwargs["parameters"][0]["value"]
    assert actual == value
    assert type(actual) is type(value)


@pytest.mark.parametrize("value", [[1, "x", None], {"nested": {"enabled": True}}, None])
def test_complex_json_parameter_values(sdk, value):
    tool = QueryCosmosDBDatabaseTool(db=CosmosDBDatabase(sdk[0], "app"))
    tool.invoke(
        {
            "container": "orders",
            "query": "SELECT VALUE @v",
            "parameters": [{"name": "@v", "value_json": json.dumps(value)}],
            "partition_key": "tenant",
        }
    )
    assert sdk[2].query_items.call_args.kwargs["parameters"] == [
        {"name": "@v", "value": value}
    ]


@pytest.mark.parametrize("value_json", ["not json", "NaN", '{"x": Infinity}'])
def test_invalid_encoded_values_are_rejected(value_json):
    with pytest.raises(ValueError, match="valid JSON"):
        QueryParameter(name="@v", value_json=value_json)
    with pytest.raises(ValueError, match="not both"):
        QueryParameter(name="@v", value=1, value_json="2")


def test_gemini_keeps_numeric_boolean_and_hierarchical_types(sdk):
    toolkit = CosmosDBDatabaseToolkit(
        db=CosmosDBDatabase(sdk[0], "app"),
        llm=FakeListLLM(responses=["ok"]),
    )
    declarations = convert_to_genai_function_declarations(toolkit.get_tools())[
        0
    ].function_declarations
    assert len(declarations) == 4
    query = next(d for d in declarations if d.name == "cosmos_db_query")
    schema = query.parameters
    parameter_value = schema.properties["parameters"].items.properties["value"]
    assert {s.type.value for s in parameter_value.any_of} == {
        "STRING",
        "INTEGER",
        "NUMBER",
        "BOOLEAN",
    }
    assert (
        schema.properties["parameters"].items.properties["value_json"].type.value
        == "STRING"
    )
    key_schema = schema.properties["partition_key"]
    key_array = next(s for s in key_schema.any_of if s.type and s.type.value == "ARRAY")
    assert {s.type.value for s in key_array.items.any_of} == {
        "STRING",
        "INTEGER",
        "NUMBER",
        "BOOLEAN",
    }


@pytest.mark.parametrize(
    "document",
    [
        {"body": "x" * 1_000_000},
        {"body": '😀"\\\n' * 10_000},
        {"values": list(range(10_000))},
        {f"field{i}": "x" * 100 for i in range(1000)},
        {"huge-key" * 10_000: "value"},
    ],
)
def test_query_output_is_bounded_valid_json_and_marked(sdk, document):
    sdk[2].query_items.return_value = iter([document])
    tool = QueryCosmosDBDatabaseTool(
        db=CosmosDBDatabase(
            sdk[0],
            "app",
            max_output_bytes=1024,
            max_string_length=100,
        )
    )
    output = tool.invoke(
        {"container": "orders", "query": "SELECT * FROM c", "partition_key": "a"}
    )
    assert len(output.encode()) <= 1024
    result = json.loads(output)
    assert result["truncated"] is True
    assert result["output_truncated"] is True
    assert result["truncation_reasons"]
    assert "items" in result and "request_charge" in result
    assert document != result["items"][0]


def test_metadata_and_discovery_outputs_are_also_bounded(sdk):
    db = CosmosDBDatabase(sdk[0], "app", max_output_bytes=1024)
    info = sdk[2].read.return_value
    info["indexingPolicy"] = {"includedPaths": [{"path": "x" * 1000}] * 100}
    result = InfoCosmosDBDatabaseTool(db=db).invoke({"container": "orders"})
    assert len(result.encode()) <= 1024
    assert json.loads(result)["output_truncated"]
    listing = db.serialize_result(["a" * 100 for _ in range(100)])
    assert len(listing.encode()) <= 1024
    assert json.loads(listing)["output_truncated"]


def test_custom_schema_does_not_require_sampling(sdk):
    db = CosmosDBDatabase(
        sdk[0], "app", container_schemas={"orders": "id: string; total: number"}
    )
    assert db.get_container_info("orders")["schema"] == "id: string; total: number"
    sdk[2].query_items.assert_not_called()


def test_sampling_uses_configured_partition_without_cross_partition_access(sdk):
    db = CosmosDBDatabase(
        sdk[0], "app", sample_documents=1, sample_partition_keys={"orders": "tenant-a"}
    )
    assert db.get_container_info("orders")["sample_documents"] == [{"id": "1"}]
    options = sdk[2].query_items.call_args.kwargs
    assert options["partition_key"] == "tenant-a"
    assert options["enable_cross_partition_query"] is False


def test_unknown_schema_and_unavailable_sampling_are_explained(sdk):
    db = CosmosDBDatabase(sdk[0], "app", sample_documents=1)
    info = db.get_container_info("orders")
    assert "schema_note" in info and "sampling_note" in info
    sdk[2].query_items.assert_not_called()


def test_http_query_diagnostics_keep_codes_and_positions_not_secrets(sdk):
    sdk[2].query_items.side_effect = CosmosHttpResponseError(
        status_code=400,
        message=json.dumps(
            {
                "errors": [
                    {
                        "code": "SC1001",
                        "location": {"start": 12, "end": 17},
                        "message": "Syntax error near secret-value",
                    }
                ],
                "request": "https://account.example.com/?sig=secret-token",
            }
        ),
    )
    tool = QueryCosmosDBDatabaseTool(db=CosmosDBDatabase(sdk[0], "app"))
    output = tool.invoke(
        {"container": "orders", "query": "SELECT * FROM c", "partition_key": "a"}
    )
    assert "SC1001" in output and "12 through 17" in output
    assert "syntax error" in output
    assert "secret" not in output and "account.example.com" not in output


@pytest.mark.parametrize(
    "error",
    [ServiceRequestError, ServiceResponseError, TimeoutError, CosmosClientTimeoutError],
)
def test_transport_failures_become_safe_tool_errors(sdk, error):
    sdk[2].query_items.side_effect = (
        error()
        if error is CosmosClientTimeoutError
        else error("sensitive connection details")
    )
    tool = QueryCosmosDBDatabaseTool(db=CosmosDBDatabase(sdk[0], "app"))
    output = tool.invoke(
        {"container": "orders", "query": "SELECT * FROM c", "partition_key": "a"}
    )
    assert "Retry later" in output and "sensitive" not in output


def test_omitted_limit_honors_small_configured_maximum(sdk):
    sdk[2].query_items.side_effect = lambda **kwargs: iter(range(20))
    db = CosmosDBDatabase(sdk[0], "app", max_results=3)
    assert len(db.query("orders", "SELECT * FROM c", partition_key="a")["items"]) == 3
    result = QueryCosmosDBDatabaseTool(db=db).invoke(
        {"container": "orders", "query": "SELECT * FROM c", "partition_key": "a"}
    )
    assert len(json.loads(result)["items"]) == 3
    with pytest.raises(ValueError, match="limit"):
        db.query("orders", "SELECT * FROM c", partition_key="a", limit=4)


def test_sdk_49_stale_pager_callback_does_not_count_toward_ru(sdk):
    def query_items(**kwargs):
        def get_next(token):
            kwargs["response_hook"](
                {"x-ms-request-charge": "2.5"}, {"Documents": [{"id": "1"}]}
            )
            return {"Documents": [{"id": "1"}]}

        pager = ItemPaged(get_next, lambda response: (None, response["Documents"]))
        kwargs["response_hook"]({"x-ms-request-charge": "99"}, pager)
        return pager

    sdk[2].query_items.side_effect = query_items
    result = CosmosDBDatabase(sdk[0], "app").query(
        "orders", "SELECT * FROM c", partition_key="a"
    )
    assert result["request_charge"] == 2.5
    assert result["items"] == [{"id": "1"}]


def test_query_through_real_sdk_proxies_counts_only_response_pages():
    connection = MagicMock()
    connection._container_properties_cache = {
        "dbs/app/colls/orders": {
            "_rid": "container-rid",
            "partitionKey": {"paths": ["/tenant"], "kind": "Hash", "version": 2},
        }
    }
    connection.last_response_headers = {"x-ms-request-charge": "99"}

    def query_items(**kwargs):
        def get_next(token):
            response = {"Documents": [{"id": "1"}]}
            kwargs["response_hook"]({"x-ms-request-charge": "3.5"}, response)
            return response

        return ItemPaged(get_next, lambda response: (None, response["Documents"]))

    connection.QueryItems.side_effect = query_items
    client = MagicMock()
    client.get_database_client.return_value = DatabaseProxy(connection, "app")
    result = CosmosDBDatabase(client, "app").query(
        "orders",
        "SELECT * FROM c",
        partition_key="a",
    )
    assert result["request_charge"] == 3.5
    assert result["items"] == [{"id": "1"}]
