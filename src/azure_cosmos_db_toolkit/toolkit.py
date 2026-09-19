"""Compose reusable Cosmos DB tools for LangChain applications."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import ConfigDict, Field

from .database import CosmosDBDatabase
from .tool import (
    InfoCosmosDBDatabaseTool,
    ListCosmosDBDatabaseTool,
    QueryCosmosDBCheckerTool,
    QueryCosmosDBDatabaseTool,
)


class CosmosDBDatabaseToolkit(BaseToolkit):
    """Build four tools from a database wrapper and a LangChain language model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    db: CosmosDBDatabase = Field(exclude=True, repr=False)
    llm: BaseLanguageModel = Field(exclude=True, repr=False)

    def get_context(self) -> dict[str, Any]:
        """Return database discovery context for an agent prompt."""
        return self.db.get_context()

    def get_tools(self) -> list[BaseTool]:
        """Return independently usable LangChain tools."""
        return [
            QueryCosmosDBDatabaseTool(db=self.db),
            InfoCosmosDBDatabaseTool(db=self.db),
            ListCosmosDBDatabaseTool(db=self.db),
            QueryCosmosDBCheckerTool(db=self.db, llm=self.llm),
        ]
