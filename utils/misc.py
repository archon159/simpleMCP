from typing import Any, Dict, List, Tuple, Optional
import re
import ast
import json

def apply_allowlist(mcp_tools: List[Any], allow: Optional[set]) -> List[Any]:
    if allow is None:
        return mcp_tools
    return [t for t in mcp_tools if t.name in allow]

def to_llm_tools(mcp_tools: List[Any], prefix: str | None) -> List[Dict[str, Any]]:
    out = []
    for t in mcp_tools:
        name = f"{prefix}.{t.name}" if prefix else t.name
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (t.description or "").strip(),
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            }
        })
    return out

def split_prefixed(name: str) -> Tuple[str, str]:
    # "brave_search.brave_web_search" -> ("brave_search", "brave_web_search")
    if "." not in name:
        raise ValueError(f"Expected prefixed tool name, got: {name}")
    return tuple(name.split(".", 1))  # (server, tool)


def extract_first_json(text: str):
    decoder = json.JSONDecoder()
    idx = text.find("{")
    if idx == -1:
        raise ValueError("No JSON found")

    obj, end = decoder.raw_decode(text[idx:].lstrip())
    return obj


def extract_tool_result_text(result_dump: dict) -> str:
    try:
        contents = result_dump.get("content", [])
        texts = []
        for c in contents:
            if isinstance(c, dict) and c.get("type") == "text":
                t = c.get("text", "")
                if t:
                    texts.append(t)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    return json.dumps(result_dump, ensure_ascii=False)

def parse_tool_call(obj: dict) -> tuple[str, dict] | None:
    if "name" in obj and "parameters" in obj and isinstance(obj["parameters"], dict):
        return obj["name"], obj["parameters"]
    return None

def parse_output(raw: str) -> tuple[str, str | None, dict | None]:
    """
    Returns:
      ("call", tool_name, tool_args)  if tool-call JSON found
      ("final", answer_text, None)    otherwise
    """
    try:
        obj = extract_first_json(raw)
        tc = parse_tool_call(obj)
        if tc is not None:
            name, params = tc
            return ("call", name, params)
    except Exception:
        if "{" in raw:
            print("Isn\'t this JSON?")  
            
    return ("final", raw.strip(), None)



def _coerce_primitive(value: Any, schema: Dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value

    sch_type = schema.get("type")

    if sch_type == "boolean":
        return value.strip().lower() in ("true", "1", "yes", "y", "on")

    if sch_type == "integer":
        try:
            return int(value.strip())
        except Exception:
            return value

    if sch_type == "number":
        try:
            return float(value.strip())
        except Exception:
            return value

    return value


def _maybe_deserialize_container(value: Any, schema: Dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value

    sch_type = schema.get("type")

    if sch_type in ("array", "object"):
        s = value.strip()
        if not (s.startswith("[") or s.startswith("{")):
            return value

        try:
            return json.loads(s)
        except Exception:
            pass

        try:
            return ast.literal_eval(s)
        except Exception:
            return value

    return value


def preprocess_by_schema(value: Any, schema: Dict[str, Any]) -> Any:
    if not isinstance(schema, dict):
        return value

    sch_type = schema.get("type")

    # object
    if sch_type == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        return {
            k: preprocess_by_schema(v, props.get(k, {}))
            for k, v in value.items()
        }

    # array
    if sch_type == "array":
        value = _maybe_deserialize_container(value, schema)
        if isinstance(value, list):
            item_schema = schema.get("items", {})
            return [preprocess_by_schema(v, item_schema) for v in value]
        return value

    # object stringified
    if sch_type == "object":
        value = _maybe_deserialize_container(value, schema)
        if isinstance(value, dict):
            return preprocess_by_schema(value, schema)
        return value

    # primitive coercion
    return _coerce_primitive(value, schema)