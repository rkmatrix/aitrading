from __future__ import annotations
from typing import Dict, Any, Iterable
import json, logging

logger = logging.getLogger(__name__)

class S3BucketProvider:
    def __init__(self, cfg: Dict[str, Any]):
        try:
            import boto3  # type: ignore
        except Exception as ex:
            raise RuntimeError("boto3 is required for S3BucketProvider") from ex
        self.boto3 = boto3
        self.id = cfg.get("id", "s3")
        self.bucket = cfg["bucket"]
        self.prefix = cfg.get("prefix", "policies/")
        self.region = cfg.get("region")
        self._s3 = boto3.client("s3", region_name=self.region)

    def fetch(self) -> Iterable[Dict[str, Any]]:
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if not key.lower().endswith((".json", ".yaml", ".yml")):
                    continue
                try:
                    body = self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8")
                    if key.lower().endswith(".json"):
                        data = json.loads(body)
                    else:
                        import yaml
                        data = yaml.safe_load(body)
                    if isinstance(data, dict):
                        data.setdefault("source_id", self.id)
                        yield data
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                item.setdefault("source_id", self.id)
                                yield item
                except Exception as ex:
                    logger.error("S3BucketProvider failed for s3://%s/%s: %s", self.bucket, key, ex)
