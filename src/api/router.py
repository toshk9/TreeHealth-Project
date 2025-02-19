from fastapi import APIRouter
from . import schemas, service


models_router = APIRouter(prefix="/models", tags=["Models"])


@models_router.post("/v1", response_model=schemas.OutputBatch)
async def post_input(input_batch: schemas.InputBatch):
    response = await service.model_inference(input_batch)
    return response
