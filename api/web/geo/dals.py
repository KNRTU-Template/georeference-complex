from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Task


class TaskDAL:
    @staticmethod
    async def create(db_session: AsyncSession, **kwargs) -> Task:
        new_task = Task(**kwargs)
        db_session.add(new_task)
        await db_session.commit()
        return new_task

    @staticmethod
    async def read(db_session: AsyncSession, **kwargs) -> list[Task]:
        stmt = select(Task).filter_by(**kwargs)
        tasks = await db_session.scalars(stmt)

        return tasks.fetchall()

    @staticmethod
    async def update(db_session: AsyncSession, task_id: int, **kwargs) -> Task:
        result = await db_session.execute(
            update(Task).where(Task.id == task_id).values(kwargs).returning(Task)
        )
        await db_session.commit()
        return result.scalar_one_or_none()
