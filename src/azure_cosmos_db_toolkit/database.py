"""Read-only operations on one Cosmos DB for NoSQL database."""

import re
from collections.abc import Sequence
from itertools import islice
from typing import Any

from azure.core.paging import ItemPaged
from azure.cosmos import CosmosClient

from ._serialization import bounded_json


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
        container_schemas: dict[str, str] | None = None,
        sample_partition_keys: dict[str, Any] | None = None,
        max_output_bytes: int = 16_384,
        max_string_length: int = 1_000,
    ) -> None:
        if not database:
            raise ValueError("database must not be empty")
        if max_results < 1:
            raise ValueError("max_results must be positive")
        if not 0 <= sample_documents <= max_results:
            raise ValueError("sample_documents must be between 0 and max_results")
        if max_output_bytes < 1024:
            raise ValueError("max_output_bytes must be at least 1024")
        if max_string_length < 1:
            raise ValueError("max_string_length must be positive")
        self._database = client.get_database_client(database)
        self._include = (
            frozenset(include_containers) if include_containers is not None else None
        )
        self._max_results = max_results
        self._sample_documents = sample_documents
        self._cross_partition = allow_cross_partition_queries
        self._schemas = dict(container_schemas or {})
        self._sample_partition_keys = dict(sample_partition_keys or {})
        self._max_output_bytes = max_output_bytes
        self._max_string_length = max_string_length

    def get_usable_container_names(self) -> list[str]:
        """Return configured names, or discover containers with database access.

        Explicit names are checked by Azure when used, not by a database-wide
        listing that container-scoped credentials may not be allowed to perform.
        """
        if self._include is not None:
            return sorted(self._include)
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
        if container in self._schemas:
            info["schema"] = self._schemas[container]
        if self._sample_documents:
            partition_key = self._sample_partition_keys.get(container)
            if partition_key is None and not self._cross_partition:
                info["sampling_note"] = (
                    "Sampling requires a configured sample_partition_keys entry "
                    "or allow_cross_partition_queries=True."
                )
            else:
                info["sample_documents"] = self.query(
                    container,
                    "SELECT TOP @count * FROM c",
                    parameters=[{"name": "@count", "value": self._sample_documents}],
                    partition_key=partition_key,
                    limit=self._sample_documents,
                )["items"]
        if "schema" not in info and "sample_documents" not in info:
            info["schema_note"] = (
                "Document fields are unknown. Configure container_schemas or "
                "document sampling before constructing queries with unknown fields."
            )
        return info

    def query(
        self,
        container: str,
        query: str,
        *,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: str | float | bool | list[Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute a SELECT with bound values and return at most ``limit`` rows.

        ``partition_key=None`` means no partition scope was supplied. Use a
        hierarchical key list where applicable. This is not a tenant boundary;
        apply authorization in the application and use read-only Azure credentials.
        """
        proxy = self._container(container)
        if limit is None:
            limit = min(10, self._max_results)
        if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE):
            raise ValueError("Only SELECT queries are supported")
        if not 1 <= limit <= self._max_results:
            raise ValueError(f"limit must be between 1 and {self._max_results}")
        if partition_key is None and not self._cross_partition:
            raise ValueError(
                "Provide a partition key; cross-partition queries are disabled"
            )
        charge = 0.0

        def record_charge(headers: dict[str, Any], response: Any) -> None:
            nonlocal charge
            # SDK 4.9 calls the hook with an unconsumed ItemPaged and stale headers
            # before it fetches any query pages. Only actual responses count.
            if isinstance(response, ItemPaged):
                return
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

    def serialize_result(self, result: Any) -> str:
        """Serialize a tool response with the configured byte and field limits."""
        return bounded_json(
            result,
            max_bytes=self._max_output_bytes,
            max_string_length=self._max_string_length,
        )
