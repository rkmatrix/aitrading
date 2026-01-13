from .local_dir import LocalDirProvider
try:
    from .http_json import HttpJsonProvider
except Exception:
    HttpJsonProvider = None
try:
    from .s3_bucket import S3BucketProvider
except Exception:
    S3BucketProvider = None

PROVIDER_MAP = {
    "local_dir": LocalDirProvider,
    "http_json": HttpJsonProvider,
    "s3_bucket": S3BucketProvider,
}
