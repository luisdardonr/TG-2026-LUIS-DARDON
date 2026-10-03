# Sistema de Reconocimiento Facial Adaptable — Código modularizado

Este es el mismo sistema que tenías en `interfaz_reconocimiento_facial.py`
(un solo archivo de ~1180 líneas), dividido en módulos por responsabilidad
y con una interfaz gráfica reorganizada. **No se cambió ninguna lógica de
reconocimiento, entrenamiento o extracción de características** — se
movió el código existente a archivos separados y se reescribió solo la
capa de interfaz (Tkinter).

## Cómo ejecutar

```bash
pip install -r requirements.txt --break-system-packages
python main.py
```

Al iniciar por primera vez, se generan los mismos archivos de siempre en
la carpeta del proyecto: `known_faces.json`, `classifier_model.pt`,
`classifier_labels.json`, `metrics_log.csv`. Si copias esos archivos (o
tu carpeta `dataset/`) desde tu versión anterior, la app los reconoce sin
problema — el formato de datos no cambió.

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Todas las constantes y umbrales (rutas, thresholds, épocas de entrenamiento). Antes estaban dispersas al inicio del archivo único. |
| `geometry.py` | `extract_geometric_features()` y todo lo relacionado con los 106 landmarks. |
| `classifier.py` | La red `FaceClassifier` (PyTorch) y las funciones de entrenar/cargar el modelo. |
| `face_store.py` | Guardar/cargar `known_faces.json`, importar dataset desde carpeta o video. |
| `metrics.py` | Registro de evaluaciones a `metrics_log.csv`. |
| `recognition_engine.py` | La clase `RecognitionEngine`: junta clasificador + comparación por coseno detrás de un único método `recognize()`. Está separada de la interfaz para que, si más adelante quieres correr una evaluación por lote (sin abrir la GUI) para el informe, puedas reutilizarla directamente. |
| `theme.py` | Paleta de colores y estilos `ttk` compartidos por toda la interfaz. |
| `gui_app.py` | La clase `FacialRecognitionApp`: arma la ventana y conecta los botones con los módulos anteriores. |
| `main.py` | Punto de entrada (`python main.py`). |
| `verificar_landmarks.py` | Sin cambios — tu herramienta de verificación visual de landmarks. |

### ¿Por qué separarlo así?

- **`geometry.py` y `classifier.py` no dependen de Tkinter.** Los podrías
  importar directamente en un script o notebook aparte para correr
  experimentos (por ejemplo, para el objetivo de "reentrenar con
  variaciones de apariencia") sin tener que abrir la interfaz gráfica.
- **`recognition_engine.py`** es el lugar único donde vive la lógica de
  "¿quién es esta persona?". Si en el futuro cambias esa lógica (otro
  umbral, otro criterio de rechazo, etc.), la cambias en un solo sitio
  y tanto la GUI como un futuro script de evaluación por lote la usan
  igual.
- **`gui_app.py`** ahora solo arma ventanas y llama funciones — no
  contiene cálculos de reconocimiento ni de entrenamiento.

## Cambios en la interfaz gráfica

- Navegación por secciones en la barra lateral izquierda (Reconocimiento,
  Comparar imágenes, Personas y dataset, Entrenamiento, Evaluación) en
  vez de tener todos los controles apilados en un solo panel.
- Paleta de colores y tipografía consistentes (`theme.py`), en vez de los
  colores planos por defecto de Tkinter.
- Nueva pestaña **"Personas y dataset"** con una tabla (`Treeview`) que
  lista cada persona registrada y cuántas capturas tiene, más un botón
  para eliminar una persona si te equivocaste al registrar.
- La cámara se inicia una sola vez al abrir la app y se mantiene activa
  en segundo plano mientras la ventana está abierta (antes se detenía
  cada vez que cambiabas de modo). Esto permite iniciar una evaluación
  desde la pestaña "Evaluación" sin tener que volver primero a la
  pestaña de video.
- El panel de "Registro de actividad" (antes a la derecha, con la misma
  información que antes mostraba `result_text`) ahora es visible sin
  importar qué sección estés viendo.

## Cambios recientes

1. **Panel de "Registro de actividad" más compacto.** Ahora arranca en
   220px (antes 300px) y se puede plegar a una franja delgada con el
   botón "—" de su encabezado; también tiene un botón 🗑 para limpiarlo.

2. **Corregido el desfase en "Comparar imágenes" con fotos de tamaños
   distintos.** Antes cada imagen se pegaba a su tamaño original y
   luego TODO el collage se forzaba a 800x400, deformando cada mitad
   de forma distinta. Ahora ambas imágenes se normalizan primero a la
   misma altura (conservando su proporción original) antes de unirlas,
   así quedan alineadas sin importar su resolución de origen.

3. **La cámara ya no se mantiene siempre encendida.** Ahora se
   enciende solo al entrar a la pestaña "Reconocimiento en vivo" (o
   automáticamente si inicias una evaluación desde la pestaña
   "Evaluación" y estaba apagada), y se apaga al salir de esa sección
   -- salvo que haya una evaluación en curso, que la mantiene activa
   hasta que la detengas. También hay un botón manual "Apagar/Encender
   cámara" dentro de "Reconocimiento" por si la quieres controlar tú
   mismo.

4. **Nueva evaluación por lote sobre dataset**, además de la
   evaluación en vivo por cámara. En la pestaña "Evaluación" ahora hay
   dos columnas: la evaluación en vivo de siempre, y una nueva
   ("Evaluación por lote (dataset)") que recorre una carpeta
   `dataset/persona/*.jpg`, reconoce cada imagen con el motor actual y
   compara contra el nombre de carpeta como identidad real -- sin
   depender de estar frente a la cámara. Ambas evaluaciones se
   guardan en el mismo `metrics_log.csv`, ahora con una columna
   `fuente` (`vivo` o `dataset`) para poder separarlas al analizar los
   resultados en el informe. Si ya tenías un `metrics_log.csv` de la
   versión anterior (sin esa columna), se migra automáticamente la
   primera vez que abras la app y se guarda una copia de respaldo
   como `metrics_log.csv.bak`.

5. **Unificado el motor de reconocimiento entre "Automático" y
   "Verificación 1 a 1".** Antes, el modo Automático usaba la
   confianza del clasificador (red neuronal, softmax -- típicamente
   cercana a 0% o 100%) mientras que elegir una persona específica
   usaba solo similitud de coseno (típicamente 60-85% incluso en
   aciertos correctos). Ambos números se mostraban igual como "%", lo
   cual parecía una inconsistencia del sistema sin serlo. Ahora, si
   hay un clasificador entrenado y la persona seleccionada es una de
   sus clases conocidas, la verificación 1 a 1 también usa la
   confianza del clasificador (evaluando específicamente la
   probabilidad de esa clase, no el argmax general) -- y solo cae de
   regreso a similitud de coseno si esa persona no está entre las
   clases del clasificador (por ejemplo, si se registró después del
   último entrenamiento). El cambio vive enteramente en
   `recognition_engine.py` (`predict_with_classifier` y `recognize`);
   `gui_app.py` solo se ajustó para que el letrero "Modo: ..." sobre
   el video muestre correctamente qué motor se está usando en cada
   caso.

## Nota importante

Como el reconocimiento y el entrenamiento no cambiaron, cualquier modelo
o dataset que ya tuvieras entrenado con la versión anterior sigue siendo
100% compatible — solo copia `known_faces.json`, `classifier_model.pt` y
`classifier_labels.json` a esta carpeta.
