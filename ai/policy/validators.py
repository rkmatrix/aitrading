from __future__ import annotations
from typing import Dict, Any, Tuple

class PolicyValidationError(Exception):
    pass

def validate_bundle(bundle: Dict[str, Any], schemas: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    missing = []
    for key in ("name", "version", "schema", "kind"):
        if key not in bundle:
            missing.append(key)
    if missing:
        raise PolicyValidationError(f"Missing required top-level fields: {missing}")

    schema_name = bundle["schema"]
    if schema_name not in schemas:
        raise PolicyValidationError(f"Unknown schema '{schema_name}'. Known: {list(schemas)}")

    required_fields = schemas[schema_name].get("required_fields", [])
    for rf in required_fields:
        if rf not in bundle:
            raise PolicyValidationError(f"Missing required field for {schema_name}: '{rf}'")

    return bundle["name"], bundle["version"]
