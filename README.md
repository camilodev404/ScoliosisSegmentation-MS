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

## Pipeline de Inferencia Esperado

El endpoint principal debe ejecutar el pipeline final construido en `ScoliosisSegmentation`:

```text
radiografia
-> modelo binario de columna
-> ROI espinal
-> modelo multiclase thoracolumbar
-> estimador de ultima vertebra visible
-> clipping anatomico
-> mascara final + vista previa + resumen JSON
```

En la version actual, la API ya define el contrato, valida la imagen, verifica artefactos y prepara la estructura de respuesta. La integracion completa del pipeline de segmentacion se portara desde el notebook final de inferencia.

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

Response actual cuando la imagen es valida y los modelos existen:

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
  "predicted_labels": [],
  "mask_path": null,
  "preview_path": null,
  "message": "Imagen recibida. Pipeline de modelos pendiente de integracion."
}
```

Response esperada cuando el pipeline completo este integrado:

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
  "predicted_labels": ["T1", "T2", "T3", "T4", "L1"],
  "mask_path": "/ruta/al/proyecto/data/results/f4a7b6e9d6a64f99a2e7f9fdd05bb857_mask.png",
  "preview_path": "/ruta/al/proyecto/data/results/f4a7b6e9d6a64f99a2e7f9fdd05bb857_preview.png",
  "message": "Inferencia completada."
}
```

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

## Notas de Versionamiento

- `artifacts/models/` contiene checkpoints versionables del microservicio.
- `data/uploads/` y `data/results/` se ignoran para evitar subir archivos de ejecucion.
- Los archivos `.gitkeep` conservan carpetas vacias en Git.

