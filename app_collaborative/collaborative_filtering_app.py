import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix, load_npz # Importamos load_npz para la matriz dispersa si fuera necesario
from surprise import Dataset, Reader, SVD
from collections import defaultdict
import os # Para verificar si los archivos existen
import re

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Recomendador de Libros Colaborativo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS personalizados (opcional pero recomendado para una mejor UI) ---
st.markdown("""
<style>
    .stApp {
        background-color: #f7eecd; /* Color de fondo similar a tu Jupyter */
        color: #333;
        font-family: 'Roboto', 'Open Sans', 'Segoe UI', 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox>div>div {
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stSlider>div>div {
        background: #ddd;
        border-radius: 5px;
    }
    .stSlider>div>div>div[data-baseweb="slider"] {
        background: #007bff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333;
        font-family: 'Roboto', 'Open Sans', 'Segoe UI', 'Arial', sans-serif;
    }
    .main-header {
        text-align: center;
        color: #4a4a4a;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .section-header {
        color: #555;
        font-size: 1.8em;
        border-bottom: 2px solid #ccc;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .book-card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        padding: 15px;
        border-radius: 8px;
        background-color: #fff; /* Fondo blanco para las tarjetas de libros */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .book-card {
        flex: 0 0 auto;
        width: 160px; /* Ancho ajustado para Streamlit */
        text-align: center;
        overflow-wrap: break-word;
        border: 1px solid #eee; /* Borde más sutil */
        padding: 10px;
        border-radius: 5px;
        background-color: #fcfcfc; /* Fondo de tarjeta un poco más claro */
        font-family: 'Roboto', sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .book-card p {
        margin: 2px 0;
        color: #555;
    }
    .book-card .title {
        font-size: 14px;
        font-weight: bold;
        height: 3em; /* Para asegurar 2 líneas */
        overflow: hidden;
        text-overflow: ellipsis;
        color: #333;
    }
    .book-card .author, .book-card .genre, .book-card .pages {
        font-size: 11px;
        color: #777;
    }
    .book-card .estimated-score {
        font-size: 12px;
        font-weight: bold;
        color: green;
        margin-top: 5px;
    }
    .avatar-card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px; /* Espacio reducido entre avatares */
        padding: 10px;
        border-radius: 8px;
        background-color: #fff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .avatar-card {
        text-align: center;
        width: 90px; /* Ancho fijo para avatar */
        overflow: hidden;
        font-family: 'Roboto', sans-serif;
        border: 1px solid #eee;
        border-radius: 5px;
        padding: 5px;
        background-color: #fcfcfc;
    }
    .avatar-card img {
        border-radius: 50%;
        border: 1px solid #ccc;
        object-fit: cover;
        display: block;
        margin: 0 auto;
    }
    .avatar-card .username {
        font-size: 10px;
        color: #555;
        margin-top: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .user-profile-summary {
        background-color: #fff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .user-profile-summary img {
        border-radius: 50%;
        border: 2px solid #ccc;
        margin-right: 20px;
    }
    .user-profile-summary h3 {
        margin: 0 0 5px 0;
        color: #007bff;
    }
    .user-profile-summary p {
        margin: 0;
        font-size: 14px;
        color: #555;
    }
    .stExpander .streamlit-expanderContent {
        padding-top: 0rem; /* Ajuste para expanders */
    }
</style>
""", unsafe_allow_html=True)

# --- Funciones de Carga de Datos y Modelos (con caché de Streamlit) ---
@st.cache_resource
def load_data_and_models():
    """Carga todos los DataFrames y modelos necesarios."""
    # Considera la estructura de tu repositorio. Si los archivos están en una subcarpeta 'data', usa './data/'
    # Si están en la raíz, usa './'
    data_path = './data/'
    models_path = './models/'

    # Ajusta las rutas si no existen las carpetas y los archivos están en la raíz
    if not os.path.exists(data_path) or not os.path.exists(os.path.join(data_path, 'df_combined_books_final.parquet')):
        data_path = './'
    if not os.path.exists(models_path) or not os.path.exists(os.path.join(models_path, 'best_svd_algo.pkl')):
        models_path = './'

    try:
        # Carga de DataFrames
        df_combined_books = pd.read_parquet(os.path.join(data_path, 'df_combined_books_final.parquet'))
        df_ratings_modified = pd.read_parquet(os.path.join(data_path, 'df_ratings_modified.parquet'))
        df_users = pd.read_parquet(os.path.join(data_path, 'df_users.parquet'))

        # Carga de Modelos
        best_svd_algo = joblib.load(os.path.join(models_path, 'best_svd_algo.pkl'))
        knn_best = joblib.load(os.path.join(models_path, 'knn_best.pkl'))

        # Carga de user_item_matrix (asumimos que es Parquet por tu confirmación)
        user_item_matrix = pd.read_parquet(os.path.join(data_path, 'user_item_matrix.parquet'))

        return df_combined_books, df_ratings_modified, df_users, best_svd_algo, user_item_matrix, knn_best

    except FileNotFoundError as e:
        st.error(f"Error: Uno o más archivos de datos/modelos no se encontraron. Asegúrate de que están en '{data_path}' o '{models_path}'. Detalles: {e}")
        st.stop() # Detiene la ejecución de la app
    except Exception as e:
        st.error(f"Error al cargar datos o modelos: {e}")
        st.stop()

# Cargar todos los recursos al inicio de la aplicación
# Estos objetos son las variables globales para tu app Streamlit
df_combined_books, df_ratings_modified, df_users, best_svd_algo, user_item_matrix, knn_best = load_data_and_models()

# --- Pre-procesamiento de géneros para Streamlit (igual que en Jupyter, crucial para el desplegable) ---
# Esto asegura que los géneros que se muestran en el desplegable estén estandarizados.
# df_combined_books ya debería venir con géneros estandarizados si se guardó bien,
# pero este paso asegura la consistencia para la generación de unique_genres_selector.
if 'genre' in df_combined_books.columns:
    df_combined_books['genre'] = df_combined_books['genre'].astype(str).replace('nan', '').replace('', 'Género Desconocido')

    def standardize_genre_name_for_app(genre_string): # Renombrada para evitar conflicto
        if pd.isna(genre_string) or not genre_string.strip():
            return "Género Desconocido"
        genres_list = [g.strip() for g in str(genre_string).replace(';', ',').split(',') if g.strip()]
        standardized_genres = []
        for g in genres_list:
            if "juvenil (young adult - ya)" in g.lower() or "young adult" in g.lower():
                standardized_genres.append("Young Adult")
            else:
                standardized_genres.append(g)
        return ', '.join(sorted(list(set(standardized_genres))))

    df_combined_books['genre'] = df_combined_books['genre'].apply(standardize_genre_name_for_app)

else:
    df_combined_books['genre'] = 'Género Desconocido'

# Generación de la lista de géneros únicos para el selector
# Aquí se utiliza explode para manejar casos donde un libro tiene múltiples géneros
all_genres_exploded = df_combined_books['genre'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()]).explode()
unique_genres = all_genres_exploded.unique().tolist()
unique_genres_selector = sorted([g for g in unique_genres if g and pd.notna(g) and g != 'Género Desconocido'])


# --- Funciones Auxiliares (adaptadas de tu Jupyter) ---

def assign_reading_preference(ratings_df_temp, books_df, unique_genres_list):
    """
    Determina el género preferido basado en las calificaciones proporcionadas.
    Adaptada para funcionar con los géneros estandarizados.
    """
    user_books_rated_now = ratings_df_temp['book_id']
    user_genres = books_df[books_df['book_id'].isin(user_books_rated_now)]['genre']

    if not user_genres.empty:
        all_rated_genres = user_genres.apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()]).explode()

        if not all_rated_genres.empty and not all_rated_genres.mode().empty:
            most_frequent_genre = all_rated_genres.mode().iloc[0]
            if pd.notna(most_frequent_genre) and most_frequent_genre.strip() != "" and most_frequent_genre != 'Género Desconocido':
                return most_frequent_genre.strip()
            elif len(all_rated_genres.mode()) > 1: # Si el más frecuente es "Desconocido", intenta el siguiente
                second_most_frequent = all_rated_genres.mode().iloc[1]
                if pd.notna(second_most_frequent) and second_most_frequent.strip() != "" and second_most_frequent != 'Género Desconocido':
                    return second_most_frequent.strip()

        valid_genres_in_list = [g for g in unique_genres_list if g and pd.notna(g) and g != 'Género Desconocido']
        if valid_genres_in_list:
            return np.random.choice(valid_genres_in_list)
        else:
            return "Preferencias desconocidas"
    else:
        valid_genres_in_list = [g for g in unique_genres_list if g and pd.notna(g) and g != 'Género Desconocido']
        if valid_genres_in_list:
            return np.random.choice(valid_genres_in_list)
        else:
            return "Preferencias desconocidas"

# Función para obtener opciones únicas de df_users (para los desplegables del perfil)
def get_unique_options(df, column):
    if column in df.columns:
        # Filtra las opciones que no sean valores por defecto o vacíos
        options = df[column].dropna().astype(str).unique().tolist()
        # Excluye valores como 'unknown', 'not specified', etc. si existen
        options = [opt for opt in options if str(opt).strip() != '' and str(opt).lower() not in ['desconocido', 'unknown', 'not specified']]
        options.sort()
        return ["--- Selecciona ---"] + options
    else:
        return ["--- No hay datos ---"]

# --- Mantenimiento del estado de la sesión para el nuevo usuario ---
if 'new_user_id' not in st.session_state:
    st.session_state.max_user_id = df_users['user_id'].max()
    st.session_state.new_user_id = st.session_state.max_user_id + 1
    st.session_state.new_user_ratings_collected = []
    st.session_state.new_user_data = {
        'user_id': st.session_state.new_user_id,
        'avatar_url': f"https://api.dicebear.com/7.x/identicon/svg?seed={st.session_state.new_user_id}",
        'username': f'NuevoUsuario_{st.session_state.new_user_id}',
        'age': '--- Selecciona ---', 'gender': '--- Selecciona ---', 'education': '--- Selecciona ---', 'country': '--- Selecciona ---',
        'reading_preference': 'Desconocida' # Se actualiza al generar recomendaciones
    }
    st.session_state.recommendations_generated = False
    st.session_state.recommendations = []
    st.session_state.similar_users_info = pd.DataFrame() # DataFrame vacío
    st.session_state.dynamic_svd_algo = None # El modelo SVD re-entrenado
    st.session_state.target_user_rated_books_set = set() # Libros calificados por el nuevo usuario
    st.session_state.recommended_book_ids_set = set() # Libros recomendados (para evitar duplicados)


# --- Funciones para mostrar libros (adaptadas para Streamlit) ---
def display_books_grid(books_details, title, include_estimation=True):
    st.markdown(f'<h3 class="section-header">{title}</h3>', unsafe_allow_html=True)
    if not books_details:
        st.info("No se encontraron libros para mostrar.")
        return

    # Usamos st.columns para crear una cuadrícula flexible
    cols_per_row = 5 # Puedes ajustar el número de columnas por fila
    num_rows = (len(books_details) + cols_per_row - 1) // cols_per_row

    # Crear los contenedores de columnas para cada fila
    # Esto es importante para que el layout se respete correctamente
    cols_containers = [st.columns(cols_per_row) for _ in range(num_rows)]

    current_col_idx = 0
    current_row_idx = 0

    for book_id, estimated_score, details in books_details:
        with cols_containers[current_row_idx][current_col_idx]:
            book_title = details.get('title', "Título desconocido")
            image_url = details.get('image_url')
            author = details.get('authors', details.get('author', "Autor desconocido"))
            genre = details.get('genre', "Género desconocido")
            pages = details.get('pages', "Páginas desconocidas")

            pages_str = "Páginas desconocidas"
            try:
                if pd.notna(pages):
                    pages_str = f"{int(pages)} págs."
            except (ValueError, TypeError):
                pass

            # Asegúrate de que la URL de la imagen sea válida o usa un placeholder
            if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.strip():
                image_url = "https://via.placeholder.com/100x150?text=No+Cover" # Placeholder

            html_card = f"""
            <div class="book-card">
                <p class="title">{book_title}</p>
                <img src="{image_url}" alt="Portada de {book_title}" width="100" height="150" style="display: block; margin: 0 auto; border: 1px solid #ccc; object-fit: cover;">
                <p class="author">Autor: {author}</p>
                <p class="genre">Género: {genre}</p>
                <p class="pages">{pages_str}</p>
                {f'<p class="estimated-score">Estimación: {estimated_score:.2f}</p>' if include_estimation else ''}
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)

        current_col_idx += 1
        if current_col_idx >= cols_per_row:
            current_col_idx = 0
            current_row_idx += 1


def display_similar_user_rated_books(similar_user_id, target_user_rated_books_set, recommended_book_ids_set,
                                     books_df, user_item_matrix_param, dynamic_svd_algo, new_user_id):
    """
    Muestra los libros calificados por un usuario similar, excluyendo los ya calificados por el nuevo usuario
    y los ya recomendados.
    """
    # Asegúrate de que el user_id esté en el índice de user_item_matrix
    if similar_user_id not in user_item_matrix_param.index:
        st.info(f"No se encontraron datos de calificación para el usuario {similar_user_id}.")
        return

    # Obtener los libros calificados por el usuario similar
    # El .T es porque KNN opera sobre filas, y user_item_matrix_param debe tener usuarios en el índice
    # y libros en las columnas
    user_ratings_sparse = user_item_matrix_param.loc[similar_user_id]
    # Filtrar solo los libros que tienen calificación (valor > 0)
    sim_user_rated_book_ids = set(user_ratings_sparse[user_ratings_sparse > 0].index)

    # Excluir libros ya calificados por el nuevo usuario o ya recomendados
    books_to_show = list(sim_user_rated_book_ids - target_user_rated_books_set - recommended_book_ids_set)

    if not books_to_show:
        st.info("Este usuario no calificó ningún libro que no tengas o que ya se te haya recomendado.")
        return

    # Limitar el número de libros a mostrar para no sobrecargar la interfaz
    display_limit = 10
    books_details_to_show_with_estimates = []

    for book_id in books_to_show:
        details = books_df[books_df['book_id'] == int(book_id)]
        if not details.empty:
            book_detail = details.iloc[0]
            try:
                # Predice la calificación que el NUEVO USUARIO daría a este libro
                pred = dynamic_svd_algo.predict(new_user_id, book_id)
                estimated_score = pred.est
            except Exception as e:
                estimated_score = np.nan # Si la predicción falla
                #st.warning(f"No se pudo predecir la calificación para el libro {book_id}: {e}") # Descomentar para depuración
            books_details_to_show_with_estimates.append((book_id, estimated_score, book_detail))
            if len(books_details_to_show_with_estimates) >= display_limit:
                break # Limita el número de libros mostrados

    books_details_to_show_with_estimates.sort(key=lambda x: x[1] if pd.notna(x[1]) else -1, reverse=True)

    display_books_grid(books_details_to_show_with_estimates, "", include_estimation=True)


# --- Título Principal de la Aplicación ---
st.markdown('<h1 class="main-header">📚 Sistema de Recomendación de Libros Colaborativo 📚</h1>', unsafe_allow_html=True)

st.write("¡Bienvenido! Para empezar a recibir recomendaciones personalizadas, dinos un poco sobre ti y califica algunos libros.")

# --- Sección de Perfil del Nuevo Usuario y Calificaciones ---
st.markdown('<h2 class="section-header">Configura tu Perfil y Califica Libros</h2>', unsafe_allow_html=True)

col_profile, col_rating = st.columns([1, 1]) # Columnas para perfil y rating

with col_profile:
    st.subheader("Datos del Nuevo Usuario")

    st.session_state.new_user_data['age'] = st.selectbox(
        'Edad:', 
        options=get_unique_options(df_users, 'age'), 
        index=0, 
        key='age_select'
    )
    st.session_state.new_user_data['gender'] = st.selectbox(
        'Género (persona):', 
        options=get_unique_options(df_users, 'gender'), 
        index=0, 
        key='gender_select'
    )
    st.session_state.new_user_data['education'] = st.selectbox(
        'Educación:', 
        options=get_unique_options(df_users, 'education'), 
        index=0, 
        key='education_select'
    )
    st.session_state.new_user_data['country'] = st.selectbox(
        'País:', 
        options=get_unique_options(df_users, 'country'), 
        index=0, 
        key='country_select'
    )

    st.markdown(f"""
    <div class="user-profile-summary">
        <img src="{st.session_state.new_user_data['avatar_url']}" alt="Avatar" width="80" height="80">
        <div>
            <h3 style="color: #007bff;">{st.session_state.new_user_data['username']} (ID: {st.session_state.new_user_id})</h3>
            <p>Edad: {st.session_state.new_user_data['age']}</p>
            <p>Género: {st.session_state.new_user_data['gender']}</p>
            <p>Educación: {st.session_state.new_user_data['education']}</p>
            <p>País: {st.session_state.new_user_data['country']}</p>
            <p>Preferencia de Lectura Estimada: <b>{st.session_state.new_user_data['reading_preference']}</b></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def format_book_option(option_tuple):
    """
    Función para formatear la visualización de una opción de libro en el selectbox.
    Recibe una tupla (label, value) y devuelve solo el label para mostrar.
    """
    if option_tuple is None or option_tuple[1] is None:
        # Esto maneja las opciones por defecto como "--- Selecciona un Género Primero ---"
        # o "--- Selecciona un Libro ---"
        return option_tuple[0] if option_tuple else "Cargando libros..."
    else:
        return option_tuple[0] # Devuelve solo el label (el primer elemento de la tupla)

with col_rating:
    st.subheader("Califica algunos libros")
    st.info("Por favor, califica al menos 3 libros para obtener mejores recomendaciones.")

    selected_genre_for_book = st.selectbox(
        'Selecciona un Género para encontrar libros:',
        options=["--- Selecciona un Género ---"] + unique_genres_selector,
        index=0,
        key='genre_select_rating'
    )

    book_options_for_genre = [] # Inicializa la lista aquí. Es importante que siempre sea una lista de tuplas.

    if selected_genre_for_book and selected_genre_for_book != "--- Selecciona un Género ---":
        filtered_books_by_genre = df_combined_books[
            df_combined_books['genre'].astype(str).str.contains(selected_genre_for_book, na=False, case=False)
        ]
        
        if not filtered_books_by_genre.empty:
            for index, row in filtered_books_by_genre.iterrows():
                # Limpiar el título: eliminar contenido entre paréntesis
                clean_title = re.sub(r'\s*\(.*?\)\s*', '', str(row['title'])).strip()
                
                # Construir la etiqueta que se mostrará en el selectbox
                book_label = f"{clean_title} por {row.get('authors', row.get('author', 'Autor desconocido'))}"
                book_options_for_genre.append((book_label, row['book_id']))
            
            # Añadir la opción por defecto al principio, como una tupla (label, value)
            book_options_for_genre.insert(0, ("--- Selecciona un Libro ---", None))
        else:
            book_options_for_genre = [("--- No se encontraron libros en este género ---", None)]
    else:
        # Si no se ha seleccionado un género, la opción por defecto es solo "--- Selecciona un Género Primero ---"
        book_options_for_genre = [("--- Selecciona un Género Primero ---", None)]


    selected_book_tuple = st.selectbox(
        'Selecciona un Libro para calificar:',
        options=book_options_for_genre, # Esta es la lista de (label, value) tuplas
        index=0,
        key='book_select_rating',
        format_func=format_book_option # <--- ¡Aquí aplicamos la función de formato!
    )

    selected_book_id = selected_book_tuple[1] if selected_book_tuple and selected_book_tuple[1] else None
    selected_book_title = selected_book_tuple[0] if selected_book_tuple and selected_book_tuple[0] else None

    rating_value = st.slider(
        'Tu Calificación (1-5 estrellas):',
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        key='rating_slider'
    )

    if st.button('Añadir Calificación', key='add_rating_button'):
        if selected_book_id is not None and rating_value is not None:
            # Comprobar si el libro ya fue calificado por este nuevo usuario
            already_rated = any(d['book_id'] == selected_book_id for d in st.session_state.new_user_ratings_collected)
            if not already_rated:
                st.session_state.new_user_ratings_collected.append({
                    'user_id': st.session_state.new_user_id,
                    'book_id': selected_book_id,
                    'rating': rating_value
                })
                st.success(f"Añadida calificación: '{selected_book_title}' con {rating_value} estrellas.")
                # Resetear las recomendaciones y el estado de generado para que se regenere al volver a pedir
                st.session_state.recommendations_generated = False 
                st.session_state.recommendations = []
                st.session_state.similar_users_info = pd.DataFrame()
            else:
                st.warning(f"Ya calificaste '{selected_book_title}'.")
        else:
            st.error("Por favor, selecciona un libro y una calificación válidos.")

    st.subheader("Tus Calificaciones Recopiladas:")
    if st.session_state.new_user_ratings_collected:
        rated_books_df_display = pd.DataFrame(st.session_state.new_user_ratings_collected)
        # Merge con df_combined_books para obtener títulos y autores para la visualización
        rated_books_display = rated_books_df_display.merge(
            df_combined_books[['book_id', 'title', 'authors']], on='book_id', how='left'
        )
        rated_books_display['Author'] = rated_books_display.apply(
            lambda row: row.get('authors', row.get('author', 'Desconocido')), axis=1
        )
        st.dataframe(
            rated_books_display[['title', 'Author', 'rating']].rename(
                columns={'title': 'Título', 'rating': 'Tu Calificación'}
            ), 
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Aún no has calificado ningún libro.")

# --- Botón para Generar Recomendaciones ---
if st.button('Generar Recomendaciones', key='generate_recommendations_main_button', type="primary", use_container_width=True):
    if len(st.session_state.new_user_ratings_collected) < 3:
        st.warning("Por favor, califica al menos 3 libros antes de generar recomendaciones.")
    else:
        with st.spinner("Generando recomendaciones y ajustando el modelo..."):
            try:
                # --- Preparar datos para re-entrenamiento SVD ---
                temp_new_user_ratings_df = pd.DataFrame(st.session_state.new_user_ratings_collected)
                # Asegúrate de que el user_id del nuevo usuario sea un número, no una cadena si es el caso
                temp_new_user_ratings_df['user_id'] = st.session_state.new_user_id

                all_current_ratings_df = pd.concat([df_ratings_modified, temp_new_user_ratings_df], ignore_index=True)

                reader = Reader(rating_scale=(1, 5))
                current_full_data = Dataset.load_from_df(all_current_ratings_df[['user_id', 'book_id', 'rating']], reader)
                current_trainset = current_full_data.build_full_trainset()

                # Re-entrenar una nueva instancia de SVD con los parámetros óptimos del modelo guardado
                svd_tuned_params = {
                    'n_factors': getattr(best_svd_algo, 'n_factors', 100),
                    'n_epochs': getattr(best_svd_algo, 'n_epochs', 20),
                    'lr_all': getattr(best_svd_algo, 'lr_all', 0.005),
                    'reg_all': getattr(best_svd_algo, 'reg_all', 0.02)
                }
                st.session_state.dynamic_svd_algo = SVD(**svd_tuned_params) # Almacenar en session_state
                st.session_state.dynamic_svd_algo.fit(current_trainset)
                st.success("Modelo de recomendación ajustado con tus preferencias.")

                # --- Calcular Preferencia de Lectura (para mostrar en perfil) ---
                st.session_state.new_user_data['reading_preference'] = assign_reading_preference(
                    temp_new_user_ratings_df, df_combined_books, unique_genres
                )

                # --- Encontrar Vecinos Similares (KNN) ---
                # Para KNN, necesitamos el vector del nuevo usuario en el mismo formato que user_item_matrix.
                # Asumiendo que user_item_matrix es un DataFrame con user_id como índice y book_id como columnas.

                # Crear una fila para el nuevo usuario en el formato de user_item_matrix
                # Asegúrate de que todas las columnas (book_id) de user_item_matrix estén presentes
                # y que las nuevas calificaciones se inserten en las columnas correctas.

                new_user_vector_data = {}
                for rated_book in st.session_state.new_user_ratings_collected:
                    new_user_vector_data[rated_book['book_id']] = rated_book['rating']

                # Crear un DataFrame de una fila para el nuevo usuario, alineando columnas con user_item_matrix
                new_user_df_for_knn = pd.DataFrame([new_user_vector_data], index=[st.session_state.new_user_id], columns=user_item_matrix.columns).fillna(0)

                # Concatenar para obtener la matriz completa para KNN (opcional, pero asegura compatibilidad)
                # Si user_item_matrix es muy grande, esto podría ser ineficiente.
                # Alternativa: pasar solo el vector del nuevo usuario a kneighbors si el modelo lo permite.
                # Considerando que tu user_item_matrix es de 7MB, es manejable.

                # Obtener los vecinos más cercanos. La entrada a kneighbors debe ser similar al conjunto de entrenamiento.
                # Si knn_best fue entrenado en csr_matrix, el vector del nuevo usuario también debe ser csr_matrix.

                # Convertir new_user_df_for_knn a csr_matrix para el modelo KNN
                # Aquí es importante que user_item_matrix_param sea el mismo tipo de objeto que se usó para entrenar knn_best
                # Si user_item_matrix.parquet es un DataFrame denso, entonces este paso es correcto.
                # Si lo entrenaste con una csr_matrix, debes convertir new_user_df_for_knn a csr_matrix.

                # Para la robustez, vamos a convertir el new_user_df_for_knn a csr_matrix antes de pasarlo al knn_best
                new_user_sparse_vector = csr_matrix(new_user_df_for_knn)
                distances, indices = knn_best.kneighbors(new_user_sparse_vector, n_neighbors=15)

                similarity_data = {'user_id': [], 'distance': []}
                for i in range(indices.shape[1]):
                    user_idx_in_matrix = indices[0][i]
                    dist = distances[0][i]
                    original_user_id = user_item_matrix.index[user_idx_in_matrix]
                    similarity_data['user_id'].append(original_user_id)
                    similarity_data['distance'].append(dist)

                similarity_df = pd.DataFrame(similarity_data)
                # Convertir distancia a similitud (para visualización, el modelo KNN usa distancia)
                if knn_best.metric == 'cosine': # Si usas 'cosine' como métrica en KNN
                    similarity_df['similarity'] = 1 - similarity_df['distance']
                else: # Para otras métricas de distancia (ej. euclidean), una inversa simple
                    similarity_df['similarity'] = 1 / (1 + similarity_df['distance'])

                st.session_state.similar_users_info = similarity_df.sort_values(by='similarity', ascending=False).head(10) # Top 10 usuarios similares

                # --- Lógica de SVD para Generar Recomendaciones de Libros ---
                st.session_state.target_user_rated_books_set = set([d['book_id'] for d in st.session_state.new_user_ratings_collected])
                all_book_ids = set(df_combined_books['book_id'].unique())
                books_to_predict_for = list(all_book_ids - st.session_state.target_user_rated_books_set)

                predictions = []
                for book_id in books_to_predict_for:
                    pred = st.session_state.dynamic_svd_algo.predict(st.session_state.new_user_id, book_id)
                    predictions.append((book_id, pred.est))

                predictions.sort(key=lambda x: x[1], reverse=True)
                top_books_svd_raw = predictions # Todas las predicciones ordenadas

                final_recommended_books_details = []
                st.session_state.recommended_book_ids_set = set() # Reinicia para nuevas recomendaciones

                num_svd_top_to_take = 5
                # 1. Tomar las N mejores recomendaciones puras de SVD
                for book_id, estimated_score in top_books_svd_raw[:num_svd_top_to_take]:
                    book_details = df_combined_books[df_combined_books['book_id'] == int(book_id)]
                    if not book_details.empty:
                        details = book_details.iloc[0]
                        final_recommended_books_details.append((book_id, estimated_score, details))
                        st.session_state.recommended_book_ids_set.add(book_id)

                # 2. Añadir recomendaciones potenciadas por género si hay espacio
                pref_genre = st.session_state.new_user_data['reading_preference']
                if pref_genre != 'Desconocida':
                    genre_boosted_candidates = []
                    for book_id, estimated_score in top_books_svd_raw:
                        if book_id not in st.session_state.recommended_book_ids_set:
                            book_details = df_combined_books[df_combined_books['book_id'] == int(book_id)]
                            if not book_details.empty:
                                details = book_details.iloc[0]
                                book_genres = str(details.get('genre', '')).split(', ')
                                if pref_genre in book_genres:
                                    genre_boosted_candidates.append((book_id, estimated_score, details))

                    genre_boosted_candidates.sort(key=lambda x: x[1], reverse=True)
                    for book_id, estimated_score, details in genre_boosted_candidates:
                        if len(final_recommended_books_details) < 10: # Si aún no tenemos 10 recomendaciones
                            final_recommended_books_details.append((book_id, estimated_score, details))
                            st.session_state.recommended_book_ids_set.add(book_id)
                        else:
                            break

                # 3. Rellenar los slots restantes con las siguientes mejores de SVD si no se alcanzan 10
                if len(final_recommended_books_details) < 10:
                    for book_id, estimated_score in top_books_svd_raw:
                        if book_id not in st.session_state.recommended_book_ids_set:
                            book_details = df_combined_books[df_combined_books['book_id'] == int(book_id)]
                            if not book_details.empty:
                                details = book_details.iloc[0]
                                final_recommended_books_details.append((book_id, estimated_score, details))
                                st.session_state.recommended_book_ids_set.add(book_id)
                                if len(final_recommended_books_details) >= 10:
                                    break

                final_recommended_books_details.sort(key=lambda x: x[1] if pd.notna(x[1]) else -1, reverse=True)
                st.session_state.recommendations = final_recommended_books_details
                st.session_state.recommendations_generated = True

            except Exception as e:
                st.error(f"Ocurrió un error al generar las recomendaciones: {e}")
                st.session_state.recommendations_generated = False
                st.session_state.recommendations = []
                st.session_state.similar_users_info = pd.DataFrame()
                st.session_state.dynamic_svd_algo = None


# --- Mostrar Recomendaciones (si se generaron) ---
if st.session_state.recommendations_generated:
    st.markdown('<h2 class="section-header">Tus Recomendaciones de Libros</h2>', unsafe_allow_html=True)
    if st.session_state.recommendations:
        display_books_grid(st.session_state.recommendations, "", include_estimation=True)
    else:
        st.info("No se pudieron generar recomendaciones. Intenta calificar más libros o revisa tus preferencias.")

    # --- Mostrar Usuarios Similares (y sus libros al hacer clic) ---
    st.markdown('<h2 class="section-header">Usuarios Similares (Haz clic en el avatar para ver sus libros)</h2>', unsafe_allow_html=True)

    if not st.session_state.similar_users_info.empty:

        # Contenedor para los avatares para que estén en una sola fila o se envuelvan bien
        st.markdown('<div class="avatar-card-container">', unsafe_allow_html=True)

        for idx, sim_user_row in st.session_state.similar_users_info.iterrows():
            sim_user_id = sim_user_row['user_id']
            sim_user_info = df_users[df_users['user_id'] == sim_user_id]

            sim_user_avatar_url = f"https://api.dicebear.com/7.x/identicon/svg?seed={sim_user_id}"
            sim_user_username = f'Usuario {sim_user_id}'

            if not sim_user_info.empty:
                sim_user_info = sim_user_info.iloc[0]
                sim_user_avatar_url = sim_user_info.get('avatar_url', sim_user_avatar_url)
                sim_user_username = sim_user_info.get('username', sim_user_username)

            # Usar st.columns para cada avatar + botón
            col_avatar = st.columns(1)[0] # Crea una columna para cada avatar individualmente
            with col_avatar:
                st.markdown(f"""
                <div class="avatar-card">
                    <img src="{sim_user_avatar_url}" alt="Avatar de {sim_user_username}" width="50" height="50">
                    <p class="username" title="{sim_user_username}">{sim_user_username}</p>
                </div>
                """, unsafe_allow_html=True)
                # El botón es "invisible" pero está ahí para el clic
                if st.button('Ver libros', key=f'view_user_books_{sim_user_id}', use_container_width=True):
                     with st.expander(f"Libros calificados por {sim_user_username}", expanded=True):
                        display_similar_user_rated_books(
                            sim_user_id,
                            st.session_state.target_user_rated_books_set, # Usar el set almacenado en session_state
                            st.session_state.recommended_book_ids_set, # Usar el set almacenado en session_state
                            df_combined_books,
                            user_item_matrix,
                            st.session_state.dynamic_svd_algo, # Usar el modelo dinámico
                            st.session_state.new_user_id
                        )
        st.markdown('</div>', unsafe_allow_html=True) # Cierra el contenedor de avatares
    else:
        st.info("No se encontraron usuarios similares para mostrar.")