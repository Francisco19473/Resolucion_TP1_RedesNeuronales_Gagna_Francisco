# Clasificación de Melanomas - Exploración de Hiperparámetros (470 Modelos)

Este repositorio contiene el código de entrenamiento, el código para visulización del dataset, las respuestas al cuestionario teórico, un informe y el registro de entrenamientos realizados tanto en MLflow (archivo .zip en el repositorio), como en Tensorboard (archivo .zip en ruta externa al repositorio).

---

## Guía de Entrega y Logs de Entrenamiento

Para facilitar la corrección y respetar las limitaciones de almacenamiento de GitHub frente a un volumen de datos tan grande, los registros se han organizado de la siguiente manera:

### 1. Monitoreo Global e Inter-Modelo (MLflow)
* **Ubicación:** mlruns.zip en este repositorio.
* **Contenido:** Contiene los hiperparámetros, métricas finales y matrices de confusión de los 470 modelos analizados (sin incluir los pesos finales de los modelos para optimizar espacio). Permite realizar el análisis comparativo directo de la búsqueda de hiperparámetros.

### 2. Historial Intra-Modelo (TensorBoard)
Debido a que la ejecución secuencial de los entrenamientos generó un único archivo binario consolidado de 1.2 GB (`events.out.tfevents`), superando las restricciones físicas de GitHub, este registro histórico se encuentra alojado en una nube externa.

* **Carpeta de Descarga Directa:** [Acceder a la carpeta en Google Drive](PON_ACA_EL_LINK_DE_TU_CARPETA_DE_DRIVE)
* **Archivo específico a descargar:** `events.out.tfevents.1780335932...`

#### Cómo visualizar los logs de TensorBoard localmente:
Si desea inspeccionar las curvas dinámicas por lote y época de los 470 modelos, siga estos pasos:

1. Descargue y descomprima el archivo `runs.zip` en la raíz de su espacio de trabajo.
2. Asegúrese de que la estructura de carpetas quede de la siguiente forma:
   ```text
   su_proyecto/
   └── runs/
       └── experimento_skin/
           └── events.out.tfevents.1780335932...
3. Desde su terminal (o el entorno de VS Code), ejecute el comando apuntando al directorio raíz de los experimentos:
   
   ```
   tensorboard --logdir=runs/experimento_skin            
   ```
   
4. Abra su navegador e ingrese a la dirección local que le indique la consola (habitualmente http://localhost:6006/).
   
