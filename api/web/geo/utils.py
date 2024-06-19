import csv
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Task
from .dals import TaskDAL
from web.config import CSV_FILENAME


class TaskManager:
    @staticmethod
    async def create(session: AsyncSession, layout_name: str, crop_name: str) -> int:
        task = await TaskDAL.create(session, status='running', layout_name=layout_name, crop_name=crop_name)
        return task.id

    @staticmethod
    async def get(session: AsyncSession, task_id: int) -> Task | None:
        task = await TaskDAL.read(session, id=task_id)
        if len(task) > 0:
            return task[0]
        return None

    @staticmethod
    async def set_error(session: AsyncSession, task_id: int):
        await TaskDAL.update(session, task_id, status='error')

    @staticmethod
    async def save_results(session: AsyncSession, task_id: int, data: dict):
        task = await TaskDAL.update(session, task_id, data=data, status='successful')
        
        with open(CSV_FILENAME, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                task.id,
                task.layout_name,
                task.crop_name,
                data['coordinates']['ul'],
                data['coordinates']['ur'],
                data['coordinates']['br'],
                data['coordinates']['bl'],
                data['crs'],
                data['start'],
                data['end'],
            ])