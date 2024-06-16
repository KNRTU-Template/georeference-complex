FROM python:3.10

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgl1

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt
RUN pip install numpy
RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"

COPY . /src

CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0"]