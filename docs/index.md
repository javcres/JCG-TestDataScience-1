# Ejercicio 1 de la prueba de Data Science

¡Hola! Bienvenido a mi propuesta de solución para el primer ejercicio de la prueba. A continuación encontrarás la documentación necesaria para poder entender el planteamiento del problema y la construcción del paquete para la puesta en producción del modelo.

## El Dataset escogido

He escogido el Dataset de clasificación [UCI Heart Disease Data (Kaggle)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/) donde tenemos que predecir la presencia de enfermedades cardiovasculares (0 = Sano 1 = Enfermo) a partir de una serie de indicadores.

*¿Por qué este Dataset?*

He escogido este Dataset porque creo que representa un problema completo de clasificación, que incluye tanto variables categóricas como variables numéricas discretas y contínuas, así como datos missing que tendremos que imputar. De esta forma el Dataset se ajustaría más a la complejidad de un problema real pero con un tamaño asequible para este ejercicio.

## Planteamiento del ejercicio

Se han seguido los siguientes pasos a la hora de abordar el problema:

1. En primer lugar realizamos un análisis exploratorio del problema, tratando de entender y visualizar la variable objetivo y los predictores de que disponemos. Como resultado de este paso, se obtendrán una serie de *insights* y visualizaciones de los datos a partir de los que determinaremos los pasos fundamentales para las tranformaciones e imputaciones de datos missing necesarios antes de seleccionar un modelo. Este paso se puede ver en el archivo `notebooks/analisis-exploratorio.ipynb`, o también embebido en esta documentación en la sección [Analisis exploratorio](analisis-exploratorio.ipynb).
2. En segundo lugar realizaremos la selección del modelo. Para ello en primer lugar crearemos una Pipeline con las transformaciones necesarias, definiremos las métricas a considerar en la evaluación del modelo, y seleccionaremos el modelo que mejor se ajuste a los datos preocupándonos también del tuning de los parámetros. Como resultado, obtendremos una Pipeline completa, desde el procesado de datos hasta la predicción, que a continuación tendremos que poner el producción. Este paso se puede ver en el archivo `notebooks/transformacion-seleccion-modelo.ipynb`, o también embebido en esta documentación en la sección [Transformacion y Seleccion de modelo](transformacion-seleccion-modelo.ipynb).
3. Por último, crearemos el código con el modelo listo para ser puesto en producción a través de un paquete de Python que nos dará la funcionalidad necesaria para entrenar el modelo y utilizarlo para predecir nuevos datos. El resultado de este paso es el repositorio de código que se proporciona donde se encuentra todo el código necesario para construir, testear, formatear y documentar el modelo en producción. Para más detalles sobre la organización del código, consultar el fichero README.md de este repositorio y para la guía de instalación de dependencias y uso del código, ver la sección [Instalación de dependencias y Uso](instalacion-y-uso.md) de esta documentación.
