# ScoliosisSegmentation-MS

Microservicio Python para servir los modelos de segmentacion de escoliosis y exponer capacidades de inferencia a una interfaz grafica.

La estrategia recomendada es mantener un endpoint principal de inferencia que ejecute todo el pipeline de la solucion y un endpoint de salud para monitoreo.

## Endpoints

### `GET /api/v1/health`

Valida que el microservicio este vivo e informa si los artefactos de modelos requeridos estan disponibles.

Respuesta esperada:

```json
{
  "status": "ok",
  "service": "ScoliosisSegmentation-MS",
  "model_ready": false,
  "missing_artifacts": [
    ".../artifacts/models/binary_spine_thoracolumbar_best.pt"
  ]
}
```

### `POST /api/v1/predict`

Endpoint principal para la interfaz grafica. Recibe una imagen de radiografia y debe ejecutar el pipeline completo:

```text
imagen
-> modelo binario de columna
-> modelo multiclase thoracolumbar
-> estimador de ultima vertebra visible
-> clipping anatomico
-> mascara final + resumen JSON
```

Entrada:

- `multipart/form-data`
- campo `image`
- formatos soportados: `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`

Salida esperada:

- identificador de prediccion;
- informacion de la imagen recibida;
- lista de vertebras detectadas;
- ruta de la mascara final;
- ruta de una vista previa con segmentacion;
- mensaje de estado.

Por ahora, el endpoint valida y guarda la imagen. Si faltan checkpoints en `artifacts/models/`, responde `503` indicando exactamente que modelos faltan. La integracion del pipeline de inferencia se portara desde el notebook final de `ScoliosisSegmentation`.

## Estructura Base

```text
ScoliosisSegmentation-MS/
├── app/
│   ├── api/              # Endpoints y routers del microservicio.
│   ├── core/             # Configuracion, settings y bootstrap de la app.
│   ├── models/           # Clases o wrappers de modelos en codigo Python.
│   ├── schemas/          # Schemas de requests/responses.
│   ├── services/         # Logica de inferencia y orquestacion del pipeline.
│   └── utils/            # Utilidades compartidas.
├── artifacts/
│   ├── models/           # Pesos/checkpoints locales. Ignorados por Git.
│   └── sample_inputs/    # Insumos de ejemplo no confidenciales.
├── data/
│   ├── uploads/          # Archivos recibidos durante ejecucion.
│   └── results/          # Resultados temporales generados por el servicio.
├── docs/                 # Documentacion tecnica del microservicio.
├── scripts/              # Scripts auxiliares de ejecucion o mantenimiento.
└── tests/                # Pruebas automatizadas.
```

## Artefactos Requeridos

Para ejecutar inferencia real, copiar localmente los checkpoints entrenados dentro de:

```text
artifacts/models/
```

Nombres esperados:

```text
binary_spine_thoracolumbar_best.pt
thoracolumbar_partial_cascade_explained_best.pt
last_visible_estimator_thoracolumbar_best.pt
```

Estos archivos estan ignorados por Git porque pueden ser pesados.

## Ejecucion Local

Instalar dependencias:

```bash
python -m pip install -e ".[dev]"
```

Levantar el servicio:

```bash
uvicorn app.main:app --reload
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

- `artifacts/models/` esta preparado para contener modelos pesados de forma local.
- `data/uploads/` y `data/results/` se ignoran para evitar versionar archivos de ejecucion.
- Los archivos `.gitkeep` permiten conservar la estructura de carpetas vacias en Git.
