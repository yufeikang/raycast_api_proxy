import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, Query, Request

from app.middleware import get_current_user_email

logger = logging.getLogger(__name__)

SYNC_DIR = os.environ.get("SYNC_DIR", "./sync")
if not os.path.exists(SYNC_DIR):
    os.makedirs(SYNC_DIR)


router = APIRouter()


@router.get("/sync", response_model=dict)
async def get_sync_data(request: Request, after: str = Query(None)):
    logger.info("Received request to /api/v1/me/sync")
    email = get_current_user_email(request)
    target = f"{SYNC_DIR}/{email}.json"
    if os.path.exists(target):
        with open(target, "r") as f:
            data = json.loads(f.read())
        if after:
            after_time = datetime.fromisoformat(after.replace("Z", "+00:00"))
            logger.info(f"Filtering data after {after_time}")
            data["updated"] = [
                item
                for item in data["updated"]
                if datetime.fromisoformat(
                    item["client_updated_at"].replace("Z", "+00:00")
                )
                > after_time
            ]
        return data
    else:
        return {"updated": [], "updated_at": None, "deleted": []}


@router.put("/sync")
async def put_sync_data(request: Request):
    email = get_current_user_email(request)
    data = await request.json()
    target = f"{SYNC_DIR}/{email}.json"
    updated_time = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    if not os.path.exists(target):
        with open(target, "w") as f:
            json.dump(
                {"updated": data["updated"], "updated_at": updated_time, "deleted": []},
                f,
            )
    else:
        with open(target, "r") as f:
            old_data = json.load(f)
        old_data["updated"].extend(data["updated"])
        old_data["updated_at"] = updated_time
        with open(target, "w") as f:
            json.dump(old_data, f)
    return {"updated_at": updated_time}
