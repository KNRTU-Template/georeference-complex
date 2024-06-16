FROM python:3.11-slim-bullseye

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY ./web /src/web
COPY ./alembic.ini /src/alembic.ini
COPY ./migrations /src/migrations

#CMD ["bash", "-c", "alembic upgrade heads && python -m web"]