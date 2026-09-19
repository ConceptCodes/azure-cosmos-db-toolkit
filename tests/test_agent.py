"""Exercise LangChain's actual agent loop without a model service or Azure."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from azure_cosmos_db_toolkit import (
    COSMOSDB_AGENT_SYSTEM_PROMPT,
    CosmosDBDatabase,
    CosmosDBDatabaseToolkit,
)


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        assert len(tools) == 4
        return self


@pytest.mark.parametrize("asynchronous", [False, True])
def test_create_agent_executes_parameterized_query(asynchronous):
    client = MagicMock()
    container = (
        client.get_database_client.return_value.get_container_client.return_value
    )
    container.query_items.return_value = iter([{"id": "order-1", "total": 25}])
    model = ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "cosmos_db_query",
                        "args": {
                            "container": "orders",
                            "query": "SELECT c.id, c.total FROM c WHERE c.status = @status",
                            "parameters": [{"name": "@status", "value": "pending"}],
                            "partition_key": "tenant-123",
                            "limit": 5,
                        },
                        "id": "query-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Order order-1 has total 25."),
        ]
    )
    toolkit = CosmosDBDatabaseToolkit(
        db=CosmosDBDatabase(client, "app", include_containers=["orders"]),
        llm=model,
    )
    agent = create_agent(
        model=model,
        tools=toolkit.get_tools(),
        system_prompt=COSMOSDB_AGENT_SYSTEM_PROMPT,
    )
    inputs = {"messages": [{"role": "user", "content": "Find pending orders."}]}
    if asynchronous:
        result = asyncio.run(agent.ainvoke(inputs, config={"recursion_limit": 10}))
    else:
        result = agent.invoke(inputs, config={"recursion_limit": 10})
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.tool_call_id == "query-1"
    assert json.loads(tool_message.content)["items"] == [{"id": "order-1", "total": 25}]
    assert result["messages"][-1].content == "Order order-1 has total 25."
    assert container.query_items.call_args.kwargs["parameters"] == [
        {"name": "@status", "value": "pending"},
    ]
