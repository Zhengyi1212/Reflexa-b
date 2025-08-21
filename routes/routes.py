# api/router.py
from fastapi import APIRouter
from . import chat, versions, merge,modify,timing

api_router = APIRouter()

# 将各个模块的路由包含进来
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(versions.router,  tags=["Version Management"])
api_router.include_router(merge.router, tags=["Code Merging"])
api_router.include_router(modify.router, tags=["Code Modification"]) 
api_router.include_router(timing.router, tags=["User Behavior"])