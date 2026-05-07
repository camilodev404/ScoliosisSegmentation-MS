# ScoliosisSegmentation-MS

Microservicio Python para servir los modelos de segmentacion de escoliosis y exponer capacidades de inferencia a una interfaz grafica.

Por el momento, este repositorio contiene la estructura base del servicio. La implementacion del API, carga de modelos y endpoints se agregara en siguientes iteraciones.

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

## Notas de Versionamiento

- `artifacts/models/` esta preparado para contener modelos pesados de forma local.
- `data/uploads/` y `data/results/` se ignoran para evitar versionar archivos de ejecucion.
- Los archivos `.gitkeep` permiten conservar la estructura de carpetas vacias en Git.
