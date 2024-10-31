# Ejercicio 1 Prueba Capgemini

> Propuesta de solución para el ejercicio de clasificación, incluyendo el análisis exploratorio, el procesado de datos y selección del modelo y la puesta en producción del mismo.

## Tabla de contenidos

1. [Documentación del proyecto](#documentación-del-proyecto)
2. [Instalacion de dependencias](#instalación-de-dependencias)
3. [Organización del repositorio](#organización-del-repositorio)
3. [Ejecución del código](#ejecución-del-código-proporcionado)

## Documentación del proyecto

La documentación con el planteamiento del ejercicio, guía de instalación y los notebooks de Exploración y Selección del Modelo se puede encontrar en [TODO](TODO), esta se ha realizado utilizando la librería [MkDocs](https://www.mkdocs.org/).

## Instalación de dependencias

El código proporcionado utiliza [Poetry](https://python-poetry.org/) como gestor de dependencias, por lo que el primer paso será [Instalar la herramienta](https://python-poetry.org/docs/#installing-with-pipx). Una vez instalada navegaremos a la carpeta raíz del proyecto y ejecutaremos el comando

`poetry install`

Con ello, ya deberíamos tener todas las dependencias para poder ejecutar tanto los jupyter notebooks como el código dentro del paquete. Se ha asegurado la reproducibilidad del modelo y los análisis mediante la definición de semillas para los casos donde interviene la aleatoriedad.

## Organización del repositorio

```plaintext
jcg-testdatascience-1
├── .github/                        # Workflows de Github
├── data/                           # Almacenamiento de datos de entrenamiento del modelo
├── dist/                           # Build del paquete final
├── docs/                           # Documentación del proyecto
├── jcg-testdatascience-1/          # Código fuente del paquete
│   ├── config/                     # Funcionalidad relativa a configuraciones y parámetros
│   │   └── config.py               # Lógica de carga de las configuraciones y parámetros
│   ├── models/                     # Almacén de modelos serializados ya entrenados
│   ├── processing/                 # Funcionalidad relativa al procesado de datos
│   │   ├── data_manager.py         # Lógica de la carga y persistencia de datos y modelos
│   │   ├── features.py             # Transformadores de datos custom
│   │   └── validation.py           # Lógica de validación de nuevos inputs
│   ├── config.yml                  # Configuraciones del modelo y el paquete
│   ├── pipeline.py                 # Definición del pipeline de predicción
│   ├── predict.py                  # Lógica de la predicción de nuevos valores
│   ├── train_pipeline.py           # Lógica del entrenamiento del modelo
│   └── VERSION                     # Versión del proyecto
├── notebooks/                      # Jupyter Notebooks
├── tests/                          # Tests del paquete
├── .gitignore                      # Archivos a ignorar por Git
├── mypy.ini                        # Configuración de MyPy
├── tox.ini                         # Configuración de Tox
├── pyproject.toml                  # Configuración del proyecto
├── mkdocs.yaml                     # Configuración del la documentación con MkDocs
└── README.md                       # README del proyecto
```

*NOTA: se ha utilizado [Este repositorio](https://github.com/trainindata/deploying-machine-learning-models/tree/master/section-05-production-model-package) como boilerplate del proyecto.*

## Ejecución del código proporcionado

Para facilitar la ejecución de tareas, se ha utilizado la herramienta [Tox](https://tox.wiki/en/4.23.2/index.html), por lo que el primer paso será [instalarla](https://tox.wiki/en/4.23.2/installation.html).

### Ejecutar entrenamiento del modelo

Si queremos ejecutar el entrenamiento del modelo utilizaremos el comando

```{shell}
tox run -e train
```

Con ello, se generará un archivo `.pkl` dentro de la carpeta `jcg_testdatascience_1/models` con el modelo entrenado.

### Ejecutar tests del código

Se han creado tests para las distintas partes del código. Para ejecutarlos, usaremos el comando

```{shell}
tox run -e test_package
```

### Ejecutar checks de formateado del código

Se han utilizado las librerías `mypy, flake8, isort` para formatear el código y adaptarlo a las convenciones. Para ejecutar estos checks, usaremos el comando

```{shell}
tox run -e checks
```

### Build del paquete final

Por último, para construir el paquete se debe utilizar el comando

```{shell}
poetry build
```

Que nos generará los archivos `.whl` y `.tar.gz` en la carpeta `/dist`
