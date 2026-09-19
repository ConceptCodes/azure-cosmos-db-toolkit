"""Bound model-facing JSON without producing invalid JSON fragments."""

import json
from typing import Any


def _encode(value: Any) -> str:
    # ASCII escaping makes the character count equal to the UTF-8 byte count.
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def bounded_json(value: Any, *, max_bytes: int, max_string_length: int) -> str:
    """Preserve query metadata and report any lossy output transformation."""
    reasons: set[str] = set()

    def clip(item: Any, budget: int, depth: int = 0) -> Any:
        if depth > 20:
            reasons.add("depth_limit")
            return None
        if isinstance(item, str):
            if len(item) > max_string_length:
                item = item[:max_string_length]
                reasons.add("field_length")
            if len(_encode(item)) > budget:
                reasons.add("output_bytes")
                low, high = 0, len(item)
                while low < high:
                    middle = (low + high + 1) // 2
                    if len(_encode(item[:middle])) <= budget:
                        low = middle
                    else:
                        high = middle - 1
                item = item[:low]
            return item
        if isinstance(item, (list, dict)):
            result: Any = [] if isinstance(item, list) else {}
            remaining = budget - 2
            entries = enumerate(item) if isinstance(item, list) else item.items()
            for key, child in entries:
                overhead = 1 if result else 0
                if isinstance(item, dict):
                    # Never shorten property names and accidentally merge keys.
                    if len(str(key)) > remaining:
                        reasons.add("output_bytes")
                        break
                    overhead += len(_encode(key)) + 1
                if remaining - overhead < 4:
                    reasons.add("output_bytes")
                    break
                clipped = clip(child, remaining - overhead, depth + 1)
                remaining -= overhead + len(_encode(clipped))
                if isinstance(result, list):
                    result.append(clipped)
                else:
                    result[key] = clipped
            return result
        if len(_encode(item)) > budget:
            reasons.add("output_bytes")
            return None
        return item

    # Reserve room for truncation flags and query accounting outside document data.
    is_query = (
        isinstance(value, dict)
        and {"items", "request_charge", "truncated"} <= value.keys()
    )
    if is_query:
        result = {**value, "items": clip(value["items"], max_bytes - 512)}
    else:
        result = clip(value, max_bytes - 512)
    if reasons:
        if not isinstance(result, dict):
            result = {"data": result}
        result["output_truncated"] = True
        result["truncation_reasons"] = sorted(reasons)
        if is_query:
            result["truncated"] = True
    return _encode(result)
