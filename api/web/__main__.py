import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from web.geo.handlers import router
from web.config import HOST, PORT

app = FastAPI(
    # openapi_url=None
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
