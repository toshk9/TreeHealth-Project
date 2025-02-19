from fastapi import FastAPI
from src.api.router import models_router


app = FastAPI()

app.include_router(models_router)
