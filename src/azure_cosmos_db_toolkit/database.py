"""Read-only operations on one Cosmos DB for NoSQL database."""

import re
from collections.abc import Sequence
from itertools import islice
from typing import Any

from azure.cosmos import CosmosClient


class CosmosDBDatabase:
    """Wrap a caller-owned client. Close the client in the calling application.

    ``include_containers=None`` exposes all containers; an empty sequence exposes
    none. Result limits bound returned data, not server-side RU consumption.
    """

    def __init__(
        self,
        client: CosmosClient,
        database: str,
        *,
        include_containers: Sequence[str] | None = None,
        max_results: int = 100,
        sample_documents: int = 0,
        allow_cross_partition_queries: bool = False,
    ) -> None:
        if not database:
            raise ValueError("database must not be empty")
        if max_results < 1:
            raise ValueError("max_results must be positive")
        if not 0 <= sample_documents <= max_results:
            raise ValueError("sample_documents must be between 0 and max_results")
        if sample_documents and not allow_cross_partition_queries:
            raise ValueError("Document sampling requires cross-partition queries")
        self._database = client.get_database_client(database)
        self._include = (
            frozenset(include_containers) if include_containers is not None else None
        )
        self._max_results = max_results
        self._sample_documents = sample_documents
        self._cross_partition = allow_cross_partition_queries

    def get_usable_container_names(self) -> list[str]:
        """List existing containers visible through this wrapper."""
        return sorted(
            item["id"]
            for item in self._database.list_containers()
            if self._include is None or item["id"] in self._include
        )

    def _container(self, name: str) -> Any:
        if self._include is not None and name not in self._include:
            raise ValueError("Container is not allowed")
        if not name:
            raise ValueError("container must not be empty")
        return self._database.get_container_client(name)

    def get_container_info(self, container: str) -> dict[str, Any]:
        """Read partition/index metadata and optionally sample documents.

        Cosmos DB is schemaless. Samples are illustrative, not a full schema.
        """
        properties = self._container(container).read()
        info = {
            "id": properties["id"],
            "partition_key": properties.get("partitionKey", {}),
            "indexing_policy": properties.get("indexingPolicy", {}),
        }
        if self._sample_documents:
            info["sample_documents"] = self.query(
                container,
                "SELECT TOP @count * FROM c",
                parameters=[{"name": "@count", "value": self._sample_documents}],
                limit=self._sample_documents,
            )["items"]
        return info

    def query(
        self,
        container: str,
        query: str,
        *,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: str | float | bool | list[Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Execute a SELECT with bound values and return at most ``limit`` rows.

        ``partition_key=None`` means no partition scope was supplied. Use a
        hierarchical key list where applicable. This is not a tenant boundary;
        apply authorization in the application and use read-only Azure credentials.
        """
        proxy = self._container(container)
        if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE):
            raise ValueError("Only SELECT queries are supported")
        if not 1 <= limit <= self._max_results:
            raise ValueError(f"limit must be between 1 and {self._max_results}")
        if partition_key is None and not self._cross_partition:
            raise ValueError(
                "Provide a partition key; cross-partition queries are disabled"
            )
        charge = 0.0

        def record_charge(headers: dict[str, Any], _: Any) -> None:
            nonlocal charge
            charge += float(headers.get("x-ms-request-charge", 0))

        options: dict[str, Any] = {
            "parameters": parameters or [],
            "max_item_count": limit,
            "response_hook": record_charge,
            "enable_cross_partition_query": self._cross_partition,
        }
        if partition_key is not None:
            options["partition_key"] = partition_key
        # max_item_count is a page size, not a total result limit.
        items = list(islice(proxy.query_items(query=query, **options), limit + 1))
        return {
            "items": items[:limit],
            "truncated": len(items) > limit,
            "request_charge": charge,
        }

    def get_context(self) -> dict[str, Any]:
        """Return discovery context suitable for an agent prompt."""
        return {"container_names": self.get_usable_container_names()}
