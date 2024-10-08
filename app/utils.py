import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Union, Optional

import httpx
import jsonpath_ng as jsonpath
import yaml
from fastapi import HTTPException

import boto3
from botocore.client import Config
import base64

logger = logging.getLogger(__name__)

RAYCAST_BACKEND = "https://backend.raycast.com"


@dataclass
class ProxyRequest:
    url: str
    method: str
    headers: dict
    body: bytes
    query_params: dict


@dataclass
class ProxyResponse:
    status_code: int
    content: bytes
    headers: dict


def load_config(file_path: str) -> Union[dict, None]:
    try:
        with open(file_path, "r") as file:
            logger.info(f"Loading config file: {file_path}")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("Config file not found at path: %s", file_path)
        return None
    except yaml.YAMLError as e:
        logger.error("Error occurred while parsing config file")
        return None
    except Exception as e:
        logger.error("Error occurred while loading config file")
        return None


def _get_mapping_key(url_path, method, type):
    return f"{url_path}:{method}:{type}".lower().removeprefix(
        "/"
    )  # remove leading slash


def init_mapping_config():
    """
    Initialize mapping config from custom_mapping.yml
    return {
        "sign_key": {
            "json_path_expr": "value"
        }
    }
    """
    mapping_config = load_config(
        os.environ.get("MAPPING_CONFIG_PATH", "custom_mapping.yml")
    )
    if mapping_config is None:
        return

    result = {}
    for path, config in mapping_config.items():
        for method, method_config in config.items():
            json_path_exprs = {}
            if method_config is None:
                continue
            # support response body only
            response = method_config.get("response")
            if response is None:
                continue
            body = response.get("body")
            if body is None:
                continue
            key = _get_mapping_key(path, method, "response:body")
            for json_path, value in body.items():
                json_path_expr = jsonpath.parse(json_path)
                json_path_exprs[json_path_expr] = value
            result[key] = json_path_exprs
    return result


MAPPING_CONFIG = init_mapping_config()


async def pass_through_request(client: httpx.AsyncClient, request: ProxyRequest):
    logger.info(f"Received request: {request.method} {request.url}")

    url = request.url

    # if host is not raycast_backend, replace it with raycast_backend
    # when use mitm tool, the host be replaced
    if not url.startswith(RAYCAST_BACKEND):
        url = url.replace(url.split("/")[2], RAYCAST_BACKEND.split("/")[2])

    if not url.startswith("https://"):
        url = url.replace("http://", "https://")

    logger.debug(f"Forwarding request to {url}")
    headers = request.headers
    # disable compression, in docker container, it will cause error, unknown reason
    headers["accept-encoding"] = "identity"
    headers["host"] = RAYCAST_BACKEND.split("/")[2]
    try:
        response = await client.request(
            request.method,
            url,
            headers=headers,
            data=request.body,
            params=request.query_params,
            timeout=60.0,
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500, detail="Error occurred while forwarding request"
        )
    content = None
    if response.content is not None:
        content = response.content
        logger.debug(
            "Response %s, status code: %s, data=%s",
            url,
            response.status_code,
            content,
        )
    filtered_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]
    response_headers = {
        key: value
        for key, value in response.headers.items()
        if key not in filtered_headers
    }
    # check and modify response content
    content = process_custom_mapping(content, request)
    return ProxyResponse(
        status_code=response.status_code, content=content, headers=response_headers
    )


def process_custom_mapping(content: bytes, request: ProxyRequest):
    if not content:
        return content
    if MAPPING_CONFIG:
        try:
            json_content = json.loads(content)
            path = request.url.removeprefix(RAYCAST_BACKEND)
            key = _get_mapping_key(path, request.method, "response:body")
            if key in MAPPING_CONFIG:
                for json_path_expr, value in MAPPING_CONFIG[key].items():
                    match = json_path_expr.find(json_content)
                    if match:
                        for match_obj in match:
                            logger.debug(f"Matched json path: {match_obj.value}")
                            match_obj.context.value[match_obj.path.fields[-1]] = value
            return json.dumps(json_content, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            logger.error("Error occurred while modifying response content")
            logger.error(e)
    return content


def json_dumps(*args, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(*args, **kwargs)

# 添加全局文件存储(后续也许可以使用数据库来替代存储？)
file_storage = {}

# Cloudflare R2 配置
CLOUDFLARE_R2_ACCESS_KEY_ID = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID")
CLOUDFLARE_R2_SECRET_ACCESS_KEY = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
CLOUDFLARE_R2_BUCKET_NAME = os.getenv("CLOUDFLARE_R2_BUCKET_NAME")
CLOUDFLARE_R2_ACCOUNT_ID = os.getenv("CLOUDFLARE_R2_ACCOUNT_ID")
CLOUDFLARE_R2_ENDPOINT = f"https://{CLOUDFLARE_R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# 创建 S3 客户端用于与 Cloudflare R2 交互
s3_client = boto3.client(
    's3',
    endpoint_url=CLOUDFLARE_R2_ENDPOINT,
    aws_access_key_id=CLOUDFLARE_R2_ACCESS_KEY_ID,
    aws_secret_access_key=CLOUDFLARE_R2_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

def get_file_info(file_id: str) -> Optional[dict]:
    """
    根据文件 ID 获取文件信息
    """
    return file_storage.get(file_id)

def generate_file_url(key: str) -> str:
    """
    使用公共域名生成文件的公开访问 URL
    """
    # 从环境变量中获取公共域名
    PUBLIC_DOMAIN = os.getenv("PUBLIC_DOMAIN")  # 请确保在 .env 文件中设置该变量

    # 构建文件 URL
    return f"https://{PUBLIC_DOMAIN}/{key}"

def store_file_info(file_id: str, file_info: dict):
    """
    存储文件信息以供后续检索
    """
    file_storage[file_id] = file_info

def generate_presigned_url(key: str, content_type: str, checksum: str) -> Optional[str]:
    """
    生成用于上传文件的预签名 URL
    """
    try:
        params = {
            'Bucket': CLOUDFLARE_R2_BUCKET_NAME,
            'Key': key,
            'ContentType': content_type,
            'ContentMD5': checksum,
        }
        url = s3_client.generate_presigned_url(
            ClientMethod='put_object',
            Params=params,
            ExpiresIn=300,
            HttpMethod='PUT'
        )
        return url
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None