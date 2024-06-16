from pydantic import BaseModel


class ResultIn(BaseModel):
    task_id: int
