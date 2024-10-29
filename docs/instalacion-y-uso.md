# Instalación y Uso del Código

El código proporcionado utiliza [Poetry](https://python-poetry.org/) como gestor de dependencias, por lo que el primer paso será [Instalar la herramienta](https://python-poetry.org/docs/#installing-with-pipx). Una vez instalada navegaremos a la carpeta raíz del proyecto y ejecutaremos el comando

`poetry install`

Con ello, ya deberíamos tener todas las dependencias para poder ejecutar tanto los jupyter notebooks como el código dentro del paquete. Se ha asegurado la reproducibilidad del modelo y los análisis mediante la definición de semillas para los casos donde interviene la aleatoriedad.

## Ejecutar entrenamiento del modelo, checks de formateado y tests del modelo

Para facilitar la ejecución de tareas, se ha utilizado la herramienta [Tox](https://tox.wiki/en/4.23.2/index.html), por lo que el primer paso será [instalarla](https://tox.wiki/en/4.23.2/installation.html).

### Ejecutar entrenamiento del modelo

Si queremos ejecutar el entrenamiento del modelo utilizaremos el comando

```{shell}
tox run -e train
```

Con ello, se generará un archivo `.pkl` dentro de la carpeta `jcg_testdatascience_1/models` con el modelo entrenado.

### Ejecutar tests del código

Se han creado tests para las distintas partes del código. Para ejecutarlos, usaremos el comando

`tox run -e test_package`

### Ejecutar checks de formateado del código

Se han utilizado las librerías `mypy, flake8, isort` para formatear el código y adaptarlo a las convenciones. Para ejecutar estos checks, usaremos el comando

`tox run -e checks`

## Build del paquete final

Por último, para construir el paquete se debe utilizar el comando

`poetry build`

Que nos generará los archivos `.whl` y `.tar.gz` en la carpeta `/dist`
