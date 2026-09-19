"""Actionable diagnostics without forwarding SDK request details to a model."""

import re

from azure.cosmos.exceptions import CosmosHttpResponseError


def describe_cosmos_error(error: CosmosHttpResponseError) -> str:
    status = error.status_code
    guidance = {
        400: "Check Cosmos SQL syntax, aliases, parameter names, and index requirements.",
        401: "Authentication failed. Ask the application owner to check credentials.",
        403: "Access denied. Check the configured container and Azure data-plane permissions.",
        404: "Database or container not found. Check the configured names.",
        408: "The request timed out. Retry later or narrow the query.",
        429: "Throughput is exhausted. Retry later or narrow the query.",
    }.get(
        status, "The service request failed. Retry later if the failure is transient."
    )
    result = f"Cosmos DB request failed (HTTP {status}). {guidance}"
    # Only extract bounded diagnostic codes/positions, never arbitrary server text,
    # request URLs, headers, or values embedded in the error message.
    if status == 400:
        message = str(error.http_error_message or "")[:16_384].replace('\\"', '"')
        codes = sorted(set(re.findall(r"\bSC\d{4}\b", message)))[:5]
        if codes:
            result += " Query diagnostics: " + ", ".join(codes) + "."
        position = re.search(
            r'"start"\s*:\s*(\d{1,8})\s*,\s*"end"\s*:\s*(\d{1,8})', message
        )
        if position:
            result += f" Check query characters {position[1]} through {position[2]}."
        if "syntax error" in message.lower():
            result += " The server reported a SQL syntax error."
        if "could not be resolved" in message.lower():
            result += (
                " An identifier could not be resolved; check aliases and field names."
            )
        if "composite index" in message.lower():
            result += " The query requires a composite index; simplify ORDER BY or configure the index."
    return result
