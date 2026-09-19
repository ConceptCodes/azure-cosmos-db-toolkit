import asyncio
import json

import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake import FakeListLLM

from azure_cosmos_db_toolkit import CosmosDBDatabase, CosmosDBDatabaseToolkit


def test_allowlist_applies_to_discovery_metadata_and_queries(sdk):
    client, database, _ = sdk
    db = CosmosDBDatabase(client, "app", include_containers=["orders"])
    assert db.get_usable_container_names() == ["orders"]
    for operation in (
        lambda: db.get_container_info("private"),
        lambda: db.query("private", "SELECT * FROM c", partition_key="a"),
    ):
        with pytest.raises(ValueError, match="not allowed"):
            operation()
    database.get_container_client.assert_not_called()
    assert CosmosDBDatabase(client, "app", include_containers=[]).get_context() == {
        "container_names": []
    }


def test_limit_is_enforced_across_pages_and_charge_is_recorded(sdk):
    client, _, container = sdk
    consumed = []

    def query_items(**kwargs):
        for page in range(3):
            kwargs["response_hook"]({"x-ms-request-charge": "2.5"}, None)
            for offset in range(2):
                consumed.append((page, offset))
                yield {"id": f"{page}-{offset}"}

    container.query_items.side_effect = query_items
    db = CosmosDBDatabase(client, "app")
    result = db.query("orders", "SELECT * FROM c", partition_key="a", limit=2)
    assert len(result["items"]) == 2
    assert result["truncated"] is True
    assert result["request_charge"] == 5.0
    assert len(consumed) == 3


@pytest.mark.parametrize("partition_key", ["tenant", 0, False, ["tenant", "region"]])
def test_parameters_and_partition_keys_pass_through(sdk, partition_key):
    client, _, container = sdk
    parameters = [{"name": "@id", "value": "' OR true"}]
    result = CosmosDBDatabase(client, "app").query(
        "orders",
        "SELECT * FROM c WHERE c.id = @id",
        parameters=parameters,
        partition_key=partition_key,
    )
    assert not result["truncated"]
    options = container.query_items.call_args.kwargs
    assert options["parameters"] == parameters
    assert options["partition_key"] == partition_key
    assert options["enable_cross_partition_query"] is False


@pytest.mark.parametrize(
    "options, message",
    [
        ({"query": "DELETE FROM c", "partition_key": "a"}, "Only SELECT"),
        ({"query": "SELECT * FROM c"}, "cross-partition"),
        ({"query": "SELECT * FROM c", "limit": 101}, "limit"),
        ({"query": "SELECT * FROM c", "limit": 0}, "limit"),
    ],
)
def test_invalid_queries_do_not_reach_sdk(sdk, options, message):
    client, _, container = sdk
    with pytest.raises(ValueError, match=message):
        CosmosDBDatabase(client, "app").query("orders", **options)
    container.query_items.assert_not_called()


def test_metadata_does_not_sample_by_default(sdk):
    client, _, container = sdk
    info = CosmosDBDatabase(client, "app").get_container_info("orders")
    assert info["partition_key"] == {"paths": ["/tenant"]}
    assert "sample_documents" not in info
    container.query_items.assert_not_called()


def test_sampling_is_opt_in(sdk):
    client, _, _ = sdk
    db = CosmosDBDatabase(
        client,
        "app",
        sample_documents=1,
        allow_cross_partition_queries=True,
    )
    assert db.get_container_info("orders")["sample_documents"] == [{"id": "1"}]


def test_tools_support_langchain_sync_async_and_checker(sdk):
    client, _, container = sdk
    toolkit = CosmosDBDatabaseToolkit(
        db=CosmosDBDatabase(client, "app", include_containers=["orders"]),
        llm=FakeListLLM(responses=["Query looks valid."]),
    )
    tools = {tool.name: tool for tool in toolkit.get_tools()}
    assert len(tools) == 4
    for tool in tools.values():
        assert tool.args_schema.model_json_schema()
    assert json.loads(tools["cosmos_db_list_containers"].invoke({})) == ["orders"]
    assert (
        json.loads(
            asyncio.run(
                tools["cosmos_db_container_info"].ainvoke(
                    {"container": "orders"},
                )
            )
        )["id"]
        == "orders"
    )
    assert tools["cosmos_db_query_checker"].invoke({"query": "SELECT * FROM c"}) == (
        "Query looks valid."
    )
    container.query_items.assert_not_called()
    result = tools["cosmos_db_query"].invoke(
        {
            "container": "orders",
            "query": "SELECT * FROM c",
            "partition_key": "a",
        }
    )
    assert json.loads(result)["items"] == [{"id": "1"}]
    assert toolkit.model_dump() == {}


def test_tool_errors_do_not_expose_sdk_details(sdk):
    client, _, container = sdk
    container.query_items.side_effect = CosmosHttpResponseError(
        status_code=400,
        message="sensitive request details",
    )
    toolkit = CosmosDBDatabaseToolkit(
        db=CosmosDBDatabase(client, "app"),
        llm=FakeListLLM(responses=["ok"]),
    )
    query_tool = next(t for t in toolkit.get_tools() if t.name == "cosmos_db_query")
    output = query_tool.invoke(
        {
            "container": "orders",
            "query": "SELECT * FROM c",
            "partition_key": "a",
        }
    )
    assert "400" in output
    assert "sensitive" not in output


@pytest.mark.parametrize("asynchronous", [False, True])
def test_checker_propagates_model_callbacks(sdk, asynchronous):
    from azure_cosmos_db_toolkit import QueryCosmosDBCheckerTool

    class Recorder(BaseCallbackHandler):
        calls = 0

        def on_llm_start(self, serialized, prompts, **kwargs):
            self.calls += 1

    recorder = Recorder()
    checker = QueryCosmosDBCheckerTool(
        db=CosmosDBDatabase(sdk[0], "app"),
        llm=FakeListLLM(responses=["Looks valid"]),
    )
    args = {"query": "SELECT * FROM c"}
    config = {"callbacks": [recorder]}
    if asynchronous:
        result = asyncio.run(checker.ainvoke(args, config=config))
    else:
        result = checker.invoke(args, config=config)
    assert result == "Looks valid"
    assert recorder.calls == 1
    assert "db" not in checker.model_dump()
    assert "llm" not in checker.model_dump()
