# Clasificación de Melanomas - Exploración de Hiperparámetros

Este repositorio contiene el código de entrenamiento, el código para visulización del dataset, los requirementes para la ejecución de los códigos mencionados, un informe, las respuestas al cuestionario teórico y el registro de entrenamientos realizados tanto en MLflow (archivo .zip en el repositorio), como en Tensorboard (archivo .zip en ruta externa al repositorio).

---

## Guía de Entrega y Logs de Entrenamiento

Para facilitar la corrección y respetar las limitaciones de almacenamiento de GitHub frente a un volumen de datos tan grande, los registros se han organizado de la siguiente manera:

### 1. Monitoreo Global e Inter-Modelo (MLflow)
* **Ubicación:** mlruns.zip (MLP) y mlruns2.zip (CNN) en este repositorio.
* **Contenido:** Contiene los hiperparámetros, métricas finales y matrices de confusión de los modelos analizados (sin incluir los pesos finales de los modelos para optimizar espacio). Permite realizar el análisis comparativo directo de la búsqueda de hiperparámetros.

### 2. Historial Intra-Modelo (TensorBoard)
Debido a que la ejecución secuencial de los entrenamientos generó archivos de gran tamaño (`events.out.tfevents`), superando las restricciones físicas de GitHub, se optó por alojar los registros históricos en una carpeta externa.

* **Carpeta de Descarga Directa:** [Acceder a la carpeta en Google Drive]([https://drive.google.com/drive/folders/13Dp60HT6Ua0aF8IMswLKWCN7pI9dX2zG?usp=sharing])


Las carpetas runs.zip y runs2.zip contiene, respectivamente, los `events.out.tfevents` del entrenamiento de la MLP y la CNN.

#### Cómo visualizar los logs de TensorBoard localmente:
Tomando de ejemplo la MLP, para inspeccionar las curvas dinámicas por lote y época de los modelos:

1. Descargar y descomprimir el archivo `runs.zip` en la raíz del espacio de trabajo.
2. Asegurar que la estructura de carpetas quede de la siguiente forma:
   ```text
   su_proyecto/
   └── runs/
       └── experimento_skin/
           └── events.out.tfevents.1780335932...
3. Desde la terminal (o el entorno de VS Code), ejecutar el comando apuntando al directorio raíz de los experimentos:
   
   ```
   tensorboard --logdir=runs/experimento_skin            
   ```
   
4. Abrir el navegador e ingrese a la dirección local que le indique la consola (habitualmente http://localhost:6006/).
   
