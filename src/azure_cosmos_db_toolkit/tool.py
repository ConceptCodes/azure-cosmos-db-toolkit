"""Individually reusable LangChain tools with explicit input schemas."""

import json
from collections.abc import Callable
from typing import Any

from azure.core.exceptions import AzureError
from azure.cosmos.exceptions import CosmosHttpResponseError
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from ._errors import describe_cosmos_error
from .database import CosmosDBDatabase
from .prompt import COSMOSDB_QUERY_CHECKER


class BaseCosmosDBDatabaseTool(BaseTool):
    """Shared database dependency and safe database error reporting."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    db: CosmosDBDatabase = Field(exclude=True, repr=False)
    handle_tool_error: bool = True

    def _call(self, operation: Callable[..., Any], **kwargs: Any) -> str:
        try:
            return self.db.serialize_result(operation(**kwargs))
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        except CosmosHttpResponseError as exc:
            raise ToolException(describe_cosmos_error(exc)) from exc
        except (AzureError, TimeoutError) as exc:
            raise ToolException(
                "Cosmos DB SDK request failed or timed out. Retry later; if it persists, "
                "ask the application owner to check connectivity and SDK timeouts."
            ) from exc


class _ListInput(BaseModel):
    tool_input: str = Field(default="", description="Leave empty.")


class ListCosmosDBDatabaseTool(BaseCosmosDBDatabaseTool):
    """Discover containers available to this toolkit."""

    name: str = "cosmos_db_list_containers"
    description: str = "List available containers before inspecting or querying them."
    args_schema: type[BaseModel] = _ListInput

    def _run(self, tool_input: str = "") -> str:
        return self._call(self.db.get_usable_container_names)


class _InfoInput(BaseModel):
    container: str = Field(min_length=1, description="An available container name.")


class InfoCosmosDBDatabaseTool(BaseCosmosDBDatabaseTool):
    """Inspect a container's partition key, indexes, and optional samples."""

    name: str = "cosmos_db_container_info"
    description: str = (
        "Inspect partition/index metadata and configured sample documents. "
        "Call cosmos_db_list_containers first. Samples are not a complete schema."
    )
    args_schema: type[BaseModel] = _InfoInput

    def _run(self, container: str) -> str:
        return self._call(self.db.get_container_info, container=container)


type JsonScalar = str | int | float | bool
type PartitionValue = JsonScalar | list[JsonScalar]


class QueryParameter(BaseModel):
    """A Cosmos SQL parameter, passed as data to the SDK."""

    name: str = Field(pattern=r"^@[A-Za-z_][A-Za-z0-9_]*$")
    value: JsonScalar | SkipJsonSchema[None] = Field(
        default=None,
        description="String, number, or Boolean parameter value. Use value_json for complex values.",
        json_schema_extra=lambda schema: schema.pop("default", None),
    )
    value_json: str = Field(
        default="",
        description='JSON-encoded array, object, or null, e.g. [1,2] or {"active":true}. Leave empty when using value.',
    )

    @model_validator(mode="after")
    def validate_value(self) -> "QueryParameter":
        if self.value_json:
            if self.value is not None:
                raise ValueError("Use either value or value_json, not both")
            try:
                json.loads(self.value_json, parse_constant=_reject_constant)
            except (ValueError, RecursionError) as exc:
                raise ValueError("value_json must contain valid JSON") from exc
        return self

    def as_sdk_parameter(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": json.loads(self.value_json) if self.value_json else self.value,
        }


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite numbers are not JSON values")


class _QueryInput(BaseModel):
    container: str = Field(min_length=1, description="Container to query.")
    query: str = Field(min_length=1, description="Cosmos SQL SELECT using @parameters.")
    parameters: list[QueryParameter] | None = None
    partition_key: PartitionValue | SkipJsonSchema[None] = Field(
        default=None,
        description="Partition value or ordered hierarchical key values. Omit for cross-partition queries.",
        json_schema_extra=lambda schema: schema.pop("default", None),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum rows; omitted uses the smaller of 10 and the configured maximum.",
    )


class QueryCosmosDBDatabaseTool(BaseCosmosDBDatabaseTool):
    """Execute a read-only Cosmos SQL query."""

    name: str = "cosmos_db_query"
    description: str = (
        "Run a Cosmos SQL SELECT with bound parameters and a partition key. "
        "Use cosmos_db_query_checker first. If a query fails, inspect metadata "
        "with cosmos_db_container_info, correct it, and retry. Returns items, "
        "truncated, and observed request_charge in RU."
    )
    args_schema: type[BaseModel] = _QueryInput

    def _run(
        self,
        container: str,
        query: str,
        parameters: list[QueryParameter] | None = None,
        partition_key: PartitionValue | None = None,
        limit: int | None = None,
    ) -> str:
        return self._call(
            self.db.query,
            container=container,
            query=query,
            parameters=[p.as_sdk_parameter() for p in parameters]
            if parameters
            else None,
            partition_key=partition_key,
            limit=limit,
        )


class _CheckerInput(BaseModel):
    query: str = Field(min_length=1, description="Cosmos SQL query to review.")


class QueryCosmosDBCheckerTool(BaseCosmosDBDatabaseTool):
    """Review SQL using an LLM without executing it."""

    llm: BaseLanguageModel = Field(exclude=True, repr=False)
    name: str = "cosmos_db_query_checker"
    description: str = (
        "Review a Cosmos SQL query before using cosmos_db_query. "
        "This is advisory and does not execute or authorize the query."
    )
    args_schema: type[BaseModel] = _CheckerInput

    def _chain(self) -> Any:
        return (
            ChatPromptTemplate.from_messages(
                [("system", COSMOSDB_QUERY_CHECKER), ("human", "{query}")]
            )
            | self.llm
            | StrOutputParser()
        )

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self._chain().invoke(
            {"query": query},
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self._chain().ainvoke(
            {"query": query},
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )
