# algo-module

## Installation and launch

1. Make directory __./layouts__ and add layout files (it's docker volumes)

2. Start application
   ```bash
   docker compose up -d --build
   ```

### How to install GDAL (if using local machine)

GDAL plays a crucial part in our approach so it's need to be installed.

https://gist.github.com/cspanring/5680334

```bash
apt install gdal-bin libgdal-dev
pip3 install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
```