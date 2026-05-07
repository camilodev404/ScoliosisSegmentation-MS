# ScoliosisSegmentation-MS

## Grupo 18

Autores de la solucion:

- Cristian Camilo Nino Rincon
- Integrante pendiente 2
- Integrante pendiente 3
- Integrante pendiente 4

## Proposito

Microservicio Python encargado de exponer la solucion de segmentacion de escoliosis a una interfaz grafica. Este repositorio corresponde a la capa de despliegue e inferencia: recibe una radiografia, ejecuta el pipeline de modelos entrenados y devuelve una respuesta consumible por un cliente web o desktop.

La solucion de investigacion y entrenamiento vive en el repositorio `ScoliosisSegmentation`. Este microservicio toma los checkpoints generados alli y los sirve mediante una API HTTP.

## Arquitectura del Microservicio

El microservicio esta organizado por capas para separar responsabilidades:

```text
Cliente grafico
   ↓ HTTP
FastAPI router
   ↓
Schemas de entrada/salida
   ↓
Servicio de inferencia
   ↓
Wrappers de modelos PyTorch
   ↓
Pipeline de segmentacion
   ↓
Mascara final + respuesta JSON
```

Estructura del repositorio:

```text
ScoliosisSegmentation-MS/
├── app/
│   ├── api/              # Endpoints y routers del microservicio.
│   ├── core/             # Configuracion, settings y rutas base.
│   ├── models/           # Clases o wrappers de modelos en codigo Python.
│   ├── schemas/          # Contratos de request/response.
│   ├── services/         # Logica de inferencia y orquestacion del pipeline.
│   └── utils/            # Utilidades compartidas.
├── artifacts/
│   ├── models/           # Checkpoints usados por el API.
│   └── sample_inputs/    # Imagenes de prueba no confidenciales.
├── data/
│   ├── uploads/          # Imagenes recibidas durante ejecucion.
│   └── results/          # Resultados generados por el servicio.
├── docs/                 # Documentacion tecnica adicional.
├── scripts/              # Scripts auxiliares.
└── tests/                # Pruebas automatizadas.
```

## Pipeline de Inferencia

El endpoint principal ejecuta el pipeline final construido en `ScoliosisSegmentation`:

```text
radiografia
-> modelo binario de columna
-> ROI espinal
-> modelo multiclase thoracolumbar
-> estimador de ultima vertebra visible
-> clipping anatomico
-> mascara final + vista previa + resumen JSON
```

La API valida la imagen, verifica que los checkpoints existan, carga los modelos de PyTorch, ejecuta la inferencia y publica los resultados generados en `data/results`.

Ademas de la vista previa visual, la respuesta incluye `vertebrae`: una salida estructurada por vertebra con caja delimitadora, centroide, area y orientacion aproximada. Este bloque es el contrato recomendado para que otro componente continue con extraccion geometrica o calculo del angulo de Cobb.

## Endpoints

### `GET /api/v1/health`

Endpoint de monitoreo. Permite saber si el servicio esta disponible y si los checkpoints requeridos existen.

Request:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Response cuando los modelos estan disponibles:

```json
{
  "status": "ok",
  "service": "ScoliosisSegmentation-MS",
  "model_ready": true,
  "missing_artifacts": []
}
```

Response cuando falta algun modelo:

```json
{
  "status": "ok",
  "service": "ScoliosisSegmentation-MS",
  "model_ready": false,
  "missing_artifacts": [
    "/ruta/al/proyecto/artifacts/models/binary_spine_thoracolumbar_best.pt"
  ]
}
```

### `POST /api/v1/predict`

Endpoint principal de inferencia. La interfaz grafica debe enviar una unica imagen de radiografia en formato `multipart/form-data`.

Request:

```bash
curl -X POST \
  -F "image=@ruta/a/radiografia.jpg" \
  http://127.0.0.1:8000/api/v1/predict
```

Entrada esperada:

- campo: `image`
- tipo: archivo de imagen
- formatos soportados: `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`

Response cuando la imagen es valida y los modelos existen:

```json
{
  "prediction_id": "f4a7b6e9d6a64f99a2e7f9fdd05bb857",
  "status": "completed",
  "image": {
    "filename": "radiografia.jpg",
    "content_type": "image/jpeg",
    "width": 1024,
    "height": 2048,
    "saved_path": "/ruta/al/proyecto/data/uploads/f4a7b6e9d6a64f99a2e7f9fdd05bb857.jpg"
  },
  "predicted_labels": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"],
  "vertebrae": [
    {
      "label": "T1",
      "mask_id": 1,
      "bbox": [121, 90, 152, 132],
      "centroid": [136.42, 111.38],
      "area_pixels": 846,
      "orientation_degrees": 84.31
    },
    {
      "label": "T2",
      "mask_id": 2,
      "bbox": [122, 133, 154, 178],
      "centroid": [137.81, 154.62],
      "area_pixels": 921,
      "orientation_degrees": 82.74
    }
  ],
  "mask_path": "/results/f4a7b6e9d6a64f99a2e7f9fdd05bb857_mask.png",
  "preview_path": "/results/f4a7b6e9d6a64f99a2e7f9fdd05bb857_preview.png",
  "message": "Inferencia completada."
}
```

Campos de `vertebrae`:

- `label`: vertebra detectada.
- `mask_id`: valor numerico de esa vertebra en la mascara multiclase.
- `bbox`: caja `[x0, y0, x1, y1]` sobre la imagen original.
- `centroid`: centroide `[x, y]` de la region segmentada.
- `area_pixels`: area de la region en pixeles.
- `orientation_degrees`: orientacion aproximada del eje principal de la region segmentada.

Response cuando faltan modelos:

```json
{
  "detail": {
    "message": "El servicio aun no tiene todos los artefactos de modelos requeridos.",
    "missing_artifacts": [
      "/ruta/al/proyecto/artifacts/models/last_visible_estimator_thoracolumbar_best.pt"
    ]
  }
}
```

Codigo HTTP:

```text
503 Service Unavailable
```

Response cuando el archivo no es valido:

```json
{
  "detail": "Formato de imagen no soportado. Usa jpg, jpeg, png, bmp, tif o tiff."
}
```

Codigo HTTP:

```text
400 Bad Request
```

## Artefactos de Modelos

Los checkpoints usados por el API se ubican en:

```text
artifacts/models/
```

Modelos requeridos:

```text
binary_spine_thoracolumbar_best.pt
thoracolumbar_partial_cascade_explained_best.pt
last_visible_estimator_thoracolumbar_best.pt
```

Tambien puede existir el checkpoint experimental:

```text
visible_range_estimator_thoracolumbar_best.pt
```

En este repositorio los modelos estan pensados para versionarse junto con el microservicio. Si GitHub muestra warnings por tamano de archivo, se puede evaluar Git LFS en una iteracion posterior.

## Ejecucion Local

Instalar dependencias:

```bash
python -m pip install -e ".[dev]"
```

Levantar el servicio:

```bash
uvicorn app.main:app --reload
```

Abrir documentacion interactiva:

```text
http://127.0.0.1:8000/docs
```

Probar salud:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Probar prediccion:

```bash
curl -X POST \
  -F "image=@ruta/a/radiografia.jpg" \
  http://127.0.0.1:8000/api/v1/predict
```

## Ejecucion con Docker

Desde este repositorio se puede construir solo el microservicio:

```bash
docker build -t scoliosis-segmentation-api .
docker run --rm -p 8000:8000 scoliosis-segmentation-api
```

Para levantar backend y frontend juntos, usar el `docker-compose.yml` ubicado en la carpeta padre `PROYECTO_SCOLIOSIS`:

```bash
cd ..
docker compose up --build
```

Servicios expuestos:

```text
API:      http://127.0.0.1:8000/api/v1/health
Frontend: http://127.0.0.1:5173
```

## Notas de Versionamiento

- `artifacts/models/` contiene checkpoints versionables del microservicio.
- `data/uploads/` y `data/results/` se ignoran para evitar subir archivos de ejecucion.
- Los archivos `.gitkeep` conservan carpetas vacias en Git.
