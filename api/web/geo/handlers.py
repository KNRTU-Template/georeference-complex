import os
import aiohttp

from fastapi import APIRouter, UploadFile, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from web.database import get_db, async_session
from .utils import TaskManager
from .schemas import ResultIn
from web.config import ALGORITHM_API_HOST

router = APIRouter(prefix='/geo', tags=[''])

STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)


async def request_process(db_session: AsyncSession, task_id: int, layout_name: str, crop_image_url: str):
    async with aiohttp.ClientSession() as http_session:
        files = {'file': open(crop_image_url, 'rb')}

        async with http_session.post(
                f'{ALGORITHM_API_HOST}api', params={'layout_name': layout_name}, data=files
        ) as response:
            if response.status != 200:
                await TaskManager.set_error(db_session, task_id)
            data = await response.json()
            await TaskManager.save_results(db_session, task_id, data)


@router.post('/')
async def start_process(
        layout_name: str,
        file: UploadFile,
        background_tasks: BackgroundTasks,
        db_session: AsyncSession = Depends(get_db)
):
    file_location = os.path.join(STORAGE_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    task_id = await TaskManager.create(db_session, layout_name, file_location)

    background_tasks.add_task(request_process, db_session, task_id, layout_name, file_location)

    return {'task_id': task_id}


@router.post('/result')
async def take_result(body: ResultIn, db_session: AsyncSession = Depends(get_db)):
    task = await TaskManager.get(db_session, body.task_id)
    if task.status == 'running':
        return {'task_id': task.id, 'status': task.status}
    elif task.status == 'successful':
        return task.data
    else:
        return {'task_id': task.id, 'status': 'error'}


if __name__ == '__main__':
    import asyncio


    async def main():
        async with async_session() as session:
            await request_process(session, 1, 'test', 'test.txt')


    asyncio.run(main())
