"""Run with COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_PARTITION_KEY,
and CHAT_MODEL set.

Install langchain and the provider package for CHAT_MODEL first. Configure that
provider's credentials separately. This example expects an orders container.
"""

import os

from azure.cosmos import CosmosClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from azure_cosmos_db_toolkit import (
    COSMOSDB_AGENT_SYSTEM_PROMPT,
    CosmosDBDatabase,
    CosmosDBDatabaseToolkit,
)


def main() -> None:
    model = init_chat_model(os.environ["CHAT_MODEL"])
    with CosmosClient(
        os.environ["COSMOS_ENDPOINT"], credential=os.environ["COSMOS_KEY"]
    ) as client:
        toolkit = CosmosDBDatabaseToolkit(
            db=CosmosDBDatabase(
                client,
                os.environ["COSMOS_DATABASE"],
                include_containers=["orders"],
                sample_documents=3,
                sample_partition_keys={"orders": os.environ["COSMOS_PARTITION_KEY"]},
            ),
            llm=model,
        )
        agent = create_agent(
            model=model,
            tools=toolkit.get_tools(),
            system_prompt=COSMOSDB_AGENT_SYSTEM_PROMPT,
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Show five recent orders in partition "
                            + os.environ["COSMOS_PARTITION_KEY"]
                        ),
                    }
                ]
            },
            config={"recursion_limit": 20},
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
