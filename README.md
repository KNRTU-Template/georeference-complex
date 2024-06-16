# Лидеры цифровой трансформации (Задача №18)

## Установка и запуск

1. Скопируйте файл переменных окружения (шаблон подготовлен для запуска без заполнения)
   ```bash
   cp .env.dist .env
   ```

2. Создайте папку **./layouts** и переместите туда файлы подложек (папка должна находиться на одном уровне с модулями **algo** и **api**)

3. Сбор и запуск контейнеров
   ```bash
   docker compose up -d --build
   ```
   
## Роуты

1. Старт анализа изображения
   ```python
   import requests

   params = {'layout_name': 'layout_20_01_2024.tif'}
   files = {'file': ('filename.tif', open('crop_0_1_0000.tif', 'rb'))}
   response = requests.post(url='http://localhost/api/geo', params=params, files=files)
   print(response.json())
   ```

2. Получение координат
   ```python
   import requests

   task_id = 1
   body = {'task_id': task_id}
   response = requests.post(url='http://localhost/api/get/result', json=body)
   print(response.json())
   ```

## Используемые порты:
 - 80, 8000, 5432