# El Sombrero Seleccionador - Análisis No Supervisado

Aplicación interactiva desarrollada en **Streamlit** que aplica técnicas de aprendizaje no supervisado (K-Means + PCA) para clasificar a los aspirantes en las casas de Hogwarts basándose en sus respuestas a un cuestionario de personalidad.

---

## Funcionalidades

- **Carga de datos** desde archivo CSV (exportado de Google Forms)
- **Cuestionario configurable**: viene precargado el de Hogwarts (16 preguntas, 4 casas), pero se puede editar o crear uno propio directamente desde la app (otras preguntas, otras opciones, hasta otro número de categorías), sin tocar código ni archivos — ver sección "Cuestionario personalizado" abajo
- **Filtros dinámicos** por género, ocupación, nivel de estrés, edad y casa dominante
- **Estadísticas básicas** (media, mediana, moda, frecuencias)
- **Entrenamiento del modelo** K-Means con visualización del método del codo y del coeficiente de silueta
- **Aplicar un modelo ya entrenado** (`.pkl`) a un CSV nuevo sin reentrenar
- **Resultados interactivos** con gráficos de PCA (con varianza explicada) y radar
- **Selección individual** con animación del Sombrero Seleccionador
- **Descarga** de datos filtrados, resultados, modelo entrenado (`.pkl`) y reporte ejecutivo (`.docx`)

---

## Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd personalidad_app_hogwarts

## Crear y Activar Entorno
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

## Instalar Dependencias
pip install -r requirements.txt

## Ejecutar la Aplicación
streamlit run app.py

## Correr las pruebas
pytest tests/ -v
```

---

## Cuestionario personalizado

Por defecto la app analiza el cuestionario de Hogwarts. Para analizar un cuestionario distinto ya no hace falta escribir ni editar ningún archivo:

1. En la barra lateral, sección "Cuestionario", elige la opción **"Editar o crear el mio"**.
2. En la parte de arriba de la página principal aparece el editor: ahí puedes cambiar el nombre del cuestionario, agregar o quitar categorías (los grupos de personalidad posibles) y agregar, editar o quitar preguntas con sus opciones de respuesta, todo con botones.
   - Cada categoría es uno de los grupos en los que puede caer una persona (por ejemplo, en vez de las 4 casas de Hogwarts podrían ser 3 estilos de liderazgo).
   - Cada pregunta necesita al menos 2 opciones de respuesta, y cada opción se asigna a una categoría desde un menú desplegable.
   - El "peso" de una pregunta es qué tanto influye en el resultado final; entre más alto, más importa esa pregunta.
3. Da clic en **"Guardar y usar este cuestionario"**. A partir de ahí, el CSV que subas debe traer una columna por cada pregunta, en el mismo orden en que quedaron en el editor — el emparejamiento respuesta→categoría es por posición y por el texto exacto de la opción elegida (ignorando mayúsculas/acentos).
4. El resto de la app (estadísticas, entrenamiento, resultados, descargas, reporte ejecutivo) funciona igual, usando las categorías y textos de tu cuestionario. Lo único que no se adapta son las tarjetas ilustradas de las 4 casas de Hogwarts (imágenes, colores, video del Sombrero), que son puramente decorativas y solo se muestran con el cuestionario por defecto.

Si prefieres trabajar con un archivo en vez del editor visual (por ejemplo para compartir el cuestionario con alguien más o guardar una copia), dentro del editor hay un apartado para descargar una copia `.json`, y en la barra lateral existe también la opción avanzada **"Subir un archivo .json"** con el mismo formato.

**Importante:** si tu CSV real ya trae el enunciado completo de cada pregunta como encabezado de columna (el comportamiento normal de un export de Google Forms), no hay ningún problema — el nombre de la columna no se usa para el cálculo, solo su posición.

