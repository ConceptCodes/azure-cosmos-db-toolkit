# Azure Cosmos DB Toolkit

A Python LangChain toolkit for **Azure Cosmos DB for NoSQL**, modeled after
`MongoDBDatabaseToolkit`.

| Tool | Purpose |
| --- | --- |
| `cosmos_db_list_containers` | Discover allowed containers |
| `cosmos_db_container_info` | Inspect partition keys, indexes, and optional samples |
| `cosmos_db_query` | Execute a parameterized Cosmos SQL SELECT |
| `cosmos_db_query_checker` | Ask an LLM to review a query without executing it |

## Install

Requires Python 3.12 or later. From this checkout:

```sh
uv sync
```

Or install into an existing environment with `pip install -e .`.
Install your preferred LangChain model provider separately.

## Usage

Construct a toolkit with an existing Cosmos client and LangChain language model:

```python
import os

from azure.cosmos import CosmosClient
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseLanguageModel

from azure_cosmos_db_toolkit import CosmosDBDatabase, CosmosDBDatabaseToolkit


def make_toolkit(client: CosmosClient, llm: BaseLanguageModel):
    db = CosmosDBDatabase(
        client,
        database=os.environ["COSMOS_DATABASE"],
        include_containers=["orders"],
        max_results=100,
        container_schemas={
            "orders": "id: string; total: number; status: string; tenant: string"
        },
    )
    return CosmosDBDatabaseToolkit(db=db, llm=llm)


# Set CHAT_MODEL to your provider:model identifier and configure its credentials.
llm = init_chat_model(os.environ["CHAT_MODEL"])
# Use a read-only key, or pass a TokenCredential with appropriate Azure RBAC.
with CosmosClient(
    os.environ["COSMOS_ENDPOINT"], credential=os.environ["COSMOS_KEY"]
) as client:
    toolkit = make_toolkit(client, llm)
    tools = toolkit.get_tools()
    query_tool = next(tool for tool in tools if tool.name == "cosmos_db_query")
    result = query_tool.invoke(
        {
            "container": "orders",
            "query": "SELECT c.id, c.total FROM c WHERE c.status = @status",
            "parameters": [{"name": "@status", "value": "pending"}],
            "partition_key": "tenant-123",
            "limit": 10,
        }
    )
    print(result)
```

Install `langchain` and your chosen model provider to run this example.
See [examples/langchain_agent.py](examples/langchain_agent.py) for a complete agent
script that keeps the client open through execution and enables document sampling.
Pass `toolkit.get_tools()` to your LangChain agent, while the client is still open:

```python
from langchain.agents import create_agent
from azure_cosmos_db_toolkit import COSMOSDB_AGENT_SYSTEM_PROMPT

with CosmosClient(
    os.environ["COSMOS_ENDPOINT"], credential=os.environ["COSMOS_KEY"]
) as client:
    toolkit = make_toolkit(client, llm)
    agent = create_agent(
        model=llm,
        tools=toolkit.get_tools(),
        system_prompt=COSMOSDB_AGENT_SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Find pending orders for tenant-123."}
            ]
        },
        config={"recursion_limit": 20},
    )
    print(result["messages"][-1].content)
```

You can also import `QueryCosmosDBDatabaseTool`, `InfoCosmosDBDatabaseTool`,
`ListCosmosDBDatabaseTool`, and `QueryCosmosDBCheckerTool` individually. Each is a
LangChain `BaseTool` with a Pydantic input schema; descriptions and names can be
customized at construction. The toolkit subclasses `BaseToolkit`, following the
[MongoDB toolkit structure](https://langchain-mongodb.readthedocs.io/en/latest/_modules/langchain_mongodb/agent_toolkit/toolkit.html).

The exported
`COSMOSDB_AGENT_SYSTEM_PROMPT` provides starting instructions, and
`toolkit.get_context()` returns accessible container names. Tools support `invoke`
and `ainvoke`; database calls run synchronous SDK work in LangChain's executor,
while the checker uses the model's async API. Callbacks propagate to the checker.
The application owns and closes the Cosmos client after tool calls finish.

## Behavior and configuration

- The wrapper uses the SDK's query API and exposes no document writes or deletes.
  Use read-only Azure credentials as the database permission boundary.
- `include_containers=None` exposes all containers in the selected database;
  an empty list exposes none. An explicit allowlist is returned without listing
  the database, so container-scoped Azure RBAC credentials work. Configured names
  are validated by Azure when accessed. Restrictions apply to metadata and queries.
- Cross-partition queries require `allow_cross_partition_queries=True`.
  Otherwise each query must supply `partition_key`. `None` means unspecified;
  explicit null partition keys are not currently supported. Hierarchical keys
  can be supplied as a list of strings, numbers, or Booleans.
- Partition selection is a query feature, **not tenant authorization**. The agent
  can choose any partition within allowed containers. Enforce user/tenant access
  in your application before exposing this toolkit.
- Supply known fields with `container_schemas={"orders": "id: string; total: number"}`.
  Otherwise enable sampling with `sample_documents=3` and
  `sample_partition_keys={"orders": "tenant-123"}`. Sampling also works without a
  partition key when cross-partition queries are enabled. If neither is available,
  metadata explains why sampling was skipped. With no schema or samples, metadata
  explicitly reports that document fields are unknown. Samples and query results
  may enter model context and configured LangChain traces. Cosmos is schemaless;
  sampled documents do not establish a complete schema.
- Queries return JSON with `items`, `truncated`, and observed `request_charge`.
  A lookahead row detects truncation. An omitted `limit` uses `min(10, max_results)`.
  RU accounting excludes the stale pre-iteration callback emitted by older SDKs.
  `max_results` limits returned rows; it is **not an RU budget**. Expensive queries can consume
  significant RU. Configure SDK timeouts, retries, and application budgets as needed.
- Database tool output is bounded by `max_output_bytes=16384` (minimum 1024),
  `max_string_length=1000`, and a nesting depth of 20. If content is shortened,
  the JSON includes `output_truncated: true` and `truncation_reasons`; query output
  also sets `truncated: true` and preserves `items` and `request_charge`.
  A shortened list response is wrapped as `{"data": [...], "output_truncated": true, ...}`.
  Narrow the projection to retrieve complete values. Direct wrapper methods return
  raw data; these serialization limits apply to database tools, not SDK traffic.
- Query checking is advisory. It does not execute queries, validate against a live
  schema, or enforce authorization. Execution remains available independently.
- Values should be passed as `@name` parameters. Tool parameters use typed scalars
  in `value` (string, number, Boolean), e.g. `{"name": "@total", "value": 25}`.
  For arrays, objects, or null, use `value_json`, e.g.
  `{"name": "@tags", "value_json": "[\"urgent\",\"new\"]"}` or
  `{"name": "@value", "value_json": "null"}`. Set only one value representation.
  Direct `db.query()` calls continue to accept native JSON values in SDK parameter
  dictionaries. SELECT-prefix validation is an
  early usability check; Cosmos DB validates the full SQL syntax.
- Expected SDK/transport failures become tool errors. HTTP errors report
  status-specific guidance; query errors retain diagnostic codes and character
  positions when available, without forwarding raw request details or secrets.

## Development

```sh
uv run pytest
uv run ruff check .
```

Tests use a mocked Cosmos SDK and a fake LLM, including synchronous and asynchronous
`create_agent` tool-call cycles and Gemini schema conversion; they require no
credentials or network. The Gemini adapter is a development dependency only.
Live Azure integration testing is still needed before production use.
