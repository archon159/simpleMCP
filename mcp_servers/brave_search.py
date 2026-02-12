import os
import re
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Annotated
from pydantic import Field
from pathlib import Path
import requests
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv
import uvicorn

load_dotenv("./secrets.env")

server_name = "brave_search"
mcp = FastMCP(server_name, json_response=True)

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip()

BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Keep enums intentionally small (representative 5).
Country = Literal["US", "GB", "DE", "KR", "JP"]
SearchLang = Literal["en", "en-gb", "de", "ko", "ja"]
SafeSearch = Literal["off", "moderate", "strict"]

# freshness supports preset values and a date range format.
FreshnessPreset = Literal["pd", "pw", "pm", "py"]

FreshnessRange = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$",
        description="Date range filter in the form YYYY-MM-DDtoYYYY-MM-DD (e.g., 2022-04-01to2022-07-30).",
    ),
]
Freshness = Optional[Annotated[
    str,
    Field(
        description="Discovery-time filter: 'pd' (24h), 'pw' (7d), 'pm' (31d), 'py' (365d), or a date range YYYY-MM-DDtoYYYY-MM-DD."
    )
]]


def _require_api_key() -> None:
    if not BRAVE_API_KEY:
        raise RuntimeError(
            "Missing BRAVE_API_KEY. Set the BRAVE_API_KEY environment variable."
        )


def _brave_get(params: Dict[str, Any]) -> Dict[str, Any]:
    _require_api_key()

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    resp = requests.get(
        BRAVE_WEB_SEARCH_ENDPOINT, headers=headers, params=params, timeout=20
    )

    try:
        payload = resp.json()
    except Exception:
        payload = {"raw_text": resp.text}

    if not resp.ok:
        raise RuntimeError(
            f"Brave API request failed: HTTP {resp.status_code} / payload={payload}"
        )

    return payload


def _validate_freshness(freshness: Optional[str]) -> Optional[str]:
    if freshness is None:
        return None

    # Allow presets.
    if freshness in ("pd", "pw", "pm", "py"):
        return freshness

    # Allow date range pattern: YYYY-MM-DDtoYYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}", freshness):
        return freshness

    # Allow the placeholder literal value (for client side error handling),
    # but reject it at runtime because it's not a real filter.
    if freshness == "YYYY-MM-DDtoYYYY-MM-DD":
        raise ValueError(
            "Invalid freshness value. Provide a real date range like 2022-04-01to2022-07-30."
        )

    raise ValueError(
        "Invalid freshness value. Use one of pd|pw|pm|py or a date range YYYY-MM-DDtoYYYY-MM-DD."
    )


def _extract_web_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    web = (payload or {}).get("web") or {}
    results = web.get("results") or []

    out: List[Dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "title": r.get("title"),
                "description": r.get("description"),
                "url": r.get("url"),
            }
        )
    return out


@mcp.tool()
def brave_web_search(
    query: Annotated[str, Field(
        max_length=400,
        description="Search query (max 400 characters)."
    )],
    count: Annotated[int, Field(
        ge=1, le=20, default=10,
        description="Number of web results to return (1-20)."
    )] = 10,
    offset: Annotated[int, Field(
        ge=0, le=9, default=0,
        description="Pagination offset (0-9)."
    )] = 0,
    country: Annotated[Country, Field(
        default="US",
        description="Country code used to localize results."
    )] = "US",
    search_lang: Annotated[SearchLang, Field(
        default="en",
        description="Language preference for results."
    )] = "en",
    safesearch: Annotated[SafeSearch, Field(
        default="moderate",
        description="Adult content filtering: off, moderate, or strict."
    )] = "moderate",
    freshness: Annotated[Optional[str], Field(
        default=None,
        description="Discovery-time filter: pd|pw|pm|py or YYYY-MM-DDtoYYYY-MM-DD."
    )] = None,
) -> List[Dict[str, Any]]:
    """
    Performs web searches using the Brave Search API.

    Returns a JSON list of web results with title, description, and URL.
    """

    # Basic runtime validation matching the typical constraints.
    if len(query) > 400:
        raise ValueError("query must be <= 400 characters.")

    count = int(count)
    offset = int(offset)

    if count < 1 or count > 20:
        raise ValueError("count must be between 1 and 20.")
    if offset < 0 or offset > 9:
        raise ValueError("offset must be between 0 and 9.")

    freshness_value = _validate_freshness(freshness)

    params: Dict[str, Any] = {
        "q": query,
        "count": count,
        "offset": offset,
        "country": country,
        "search_lang": search_lang,
        "safesearch": safesearch,
        # Unused options
        "spellcheck": True,
        "text_decorations": True,
        "result_filter": ["web", "query"],
    }

    if freshness_value is not None:
        params["freshness"] = freshness_value

    payload = _brave_get(params)
    return _extract_web_results(payload)


def run_server(host: str = "127.0.0.1", port: int = 8001):
    print(f"[{server_name}] starting on {host}:{port}")

    app = mcp.streamable_http_app()
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()