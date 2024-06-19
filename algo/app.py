from fastapi import FastAPI, UploadFile
from osgeo import gdal
import io
from georeferencing import find_coordinates

app = FastAPI()


@app.post("/api")
async def add_task(layout_name: str, file: UploadFile):
    content = await file.read()
    file_bytes = io.BytesIO(content)

    vsi_filename = f'/vsimem/{file.filename}'

    gdal.FileFromMemBuffer(vsi_filename, file_bytes.getvalue())

    crop_arr = gdal.Open(vsi_filename).ReadAsArray()[:3, :, :]

    layout = gdal.Open(f"layouts/{layout_name}")

    geo_corners, start, end = find_coordinates(layout, crop_arr)

    geo_corners = geo_corners.tolist()

    coordinates = {
        "ul": f'{geo_corners[0][0]};{geo_corners[0][1]}',
        "ur": f'{geo_corners[1][0]};{geo_corners[1][1]}',
        "br": f'{geo_corners[2][0]};{geo_corners[2][1]}',
        "bl": f'{geo_corners[3][0]};{geo_corners[3][1]}',
    }
    crs = "EPSG:32637"

    gdal.Unlink(vsi_filename)

    return {
        "layout_name": layout_name,
        "file_name": file.filename,
        "coordinates": coordinates,
        "crs": crs,
        "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%S")
    }
