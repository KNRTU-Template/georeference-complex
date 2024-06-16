from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB

from web.database import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    status = Column(String)
    layout_name = Column(String)
    crop_name = Column(String)
    data = Column(JSONB)
    processed_image = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
