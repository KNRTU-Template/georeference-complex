import csv
import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path

from web.geo.handlers import router
from web.config import HOST, PORT

from web.config import CSV_FILENAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    csv_file = Path(CSV_FILENAME)

    if not csv_file.exists():
        writer = csv.writer(open(CSV_FILENAME, 'a'))
        header = ["task_id", "layout_name", "file_name", "ul", "ur", "br", "bl", "crs", "start", "end"]
        writer.writerow(header)

    yield


app = FastAPI(
    lifespan=lifespan,  # openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix='/api')

api_router.include_router(router)

app.include_router(api_router)

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
