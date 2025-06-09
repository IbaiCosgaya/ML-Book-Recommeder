import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix, load_npz # Importamos load_npz para la matriz dispersa si fuera necesario
from surprise import Dataset, Reader, SVD
from collections import defaultdict
import os # Para verificar si los archivos existen
import re # Asegúrate de que esta importación esté presente

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="DeLibreroo - Recomendador de Libros",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS personalizados (mejorados) ---
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
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
    }
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
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
    .similar-users-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        padding: 15px;
        border-radius: 8px;
        background-color: #fff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        justify-content: flex-start;
        align-items: stretch;
    }
    .avatar-card {
        text-align: center;
        width: 120px;
        padding: 15px;
        border: 1px solid #eee;
        border-radius: 8px;
        background-color: #fcfcfc;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .avatar-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .avatar-card img {
        border-radius: 50%;
        border: 2px solid #ccc;
        object-fit: cover;
        display: block;
        margin: 0 auto 10px auto;
    }
    .avatar-card .username {
        font-size: 12px;
        color: #555;
        font-weight: bold;
        text-align: center;
        word-wrap: break-word;
    }
    .avatar-card .similarity {
        font-size: 10px;
        color: #007bff;
        margin-top: 5px;
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
    .book-preview {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #fff;
        margin: 10px 0;
    }
    .book-preview img {
        border-radius: 4px;
        border: 1px solid #ccc;
        object-fit: cover;
    }
    .book-preview-info {
        flex: 1;
    }
    .book-preview-info h4 {
        margin: 0 0 5px 0;
        color: #333;
        font-size: 16px;
    }
    .book-preview-info p {
        margin: 2px 0;
        font-size: 14px;
        color: #666;
    }
    .stExpander .streamlit-expanderContent {
        padding-top: 0rem; /* Ajuste para expanders */
    }
    .horizontal-scroll {
        display: flex;
        overflow-x: auto;
        gap: 15px;
        padding: 15px;
        border-radius: 8px;
        background-color: #fff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .horizontal-scroll::-webkit-scrollbar {
        height: 8px;
    }
    .horizontal-scroll::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .horizontal-scroll::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .horizontal-scroll::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- Funciones de Carga de Datos y Modelos (con caché de Streamlit) ---
@st.cache_resource
def load_data_and_models():

    df_combined_books_path = 'df_combined_books_final.parquet'
    ratings_path = 'app_collaborative/df_ratings_modified.parquet'
    users_path = 'app_collaborative/df_users.parquet'
    svd_path = 'app_collaborative/best_svd_algo.pkl'
    knn_path = 'app_collaborative/knn_best.pkl'
    user_item_matrix_path = 'app_collaborative/user_item_matrix.parquet'

    try:
        df_combined_books = pd.read_parquet(df_combined_books_path)
        df_ratings_modified = pd.read_parquet(ratings_path)
        df_users = pd.read_parquet(users_path)
        best_svd_algo = joblib.load(svd_path)
        knn_best = joblib.load(knn_path)
        user_item_matrix = pd.read_parquet(user_item_matrix_path)
        
        return df_combined_books, df_ratings_modified, df_users, best_svd_algo, user_item_matrix, knn_best
    except FileNotFoundError as e:
        # Este error es crucial para la depuración en Streamlit Cloud
        st.error(f"Error: Uno o más archivos de datos/modelos no se encontraron. Asegúrate de que las rutas son correctas y los archivos existen. Detalles: {e}")
        st.stop() # Detiene la ejecución de la app de forma controlada
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
    if title:
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

def display_book_preview(book_details):
    """Muestra una vista previa del libro seleccionado"""
    if book_details is None or book_details.empty:
        return
        
    book = book_details.iloc[0] if isinstance(book_details, pd.DataFrame) else book_details
    
    book_title = book.get('title', "Título desconocido")
    image_url = book.get('image_url')
    author = book.get('authors', book.get('author', "Autor desconocido"))
    genre = book.get('genre', "Género desconocido")
    pages = book.get('pages', "Páginas desconocidas")
        
    pages_str = "Páginas desconocidas"
    try:
        if pd.notna(pages):
            pages_str = f"{int(pages)} páginas"
    except (ValueError, TypeError):
        pass
        
    # Asegúrate de que la URL de la imagen sea válida o usa un placeholder
    if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.strip():
        image_url = "https://via.placeholder.com/80x120?text=No+Cover"
        
    html_preview = f"""
    <div class="book-preview">
        <img src="{image_url}" alt="Portada de {book_title}" width="80" height="120">
        <div class="book-preview-info">
            <h4>{book_title}</h4>
            <p><strong>Autor:</strong> {author}</p>
            <p><strong>Género:</strong> {genre}</p>
            <p><strong>Páginas:</strong> {pages_str}</p>
        </div>
    </div>
    """
    st.markdown(html_preview, unsafe_allow_html=True)

# --- Título Principal de la Aplicación con Logo ---
# Intentar mostrar el logo si existe
logo_path = "./DeLibreroo.png"
if os.path.exists(logo_path):
    col_logo, col_title = st.columns([1, 3])
    with col_logo:
        st.image(logo_path, width=150)
    with col_title:
        st.markdown('<h1 class="main-header">Sistema de Recomendación de Libros Colaborativo</h1>', unsafe_allow_html=True)
else:
    # Si no existe el logo, buscar en diferentes ubicaciones posibles
    possible_paths = ["./logo/DeLibreroo.png", "./images/DeLibreroo.png", "./assets/DeLibreroo.png", "DeLibreroo.png"]
    logo_found = False
        
    for path in possible_paths:
        if os.path.exists(path):
            col_logo, col_title = st.columns([1, 3])
            with col_logo:
                st.image(path, width=150)
            with col_title:
                st.markdown('<h1 class="main-header">Sistema de Recomendación de Libros Colaborativo</h1>', unsafe_allow_html=True)
            logo_found = True
            break
        
    if not logo_found:
        st.markdown('<h1 class="main-header">📚 DeLibreroo - Sistema de Recomendación de Libros 📚</h1>', unsafe_allow_html=True)
        st.info("💡 Para mostrar el logo, coloca el archivo 'DeLibreroo.png' en la carpeta raíz de tu aplicación.")

st.write("¡Bienvenido a DeLibreroo! Para empezar a recibir recomendaciones personalizadas, dinos un poco sobre ti y califica algunos libros.")

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

# --- Función completa para formatear las opciones del selectbox ---
def format_book_option(option_tuple):
    """
    Función para formatear la visualización de una opción de libro en el selectbox.
    Recibe una tupla (label, value) y devuelve solo el label para mostrar.
    """
    if option_tuple is None or option_tuple[1] is None:
        # Esto maneja las opciones por defecto como "--- Selecciona un Género Primero ---"
        # o "--- Selecciona un Libro ---", asegurando que el label sea el texto.
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
        
        # ¡CORRECCIÓN de NameError aquí!
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
        format_func=format_book_option # ¡Aplicamos la función de formato aquí!
    )

    # Solo muestra la vista previa del libro si se ha seleccionado un libro válido
    if selected_book_tuple and selected_book_tuple[1] is not None:
        selected_book_id = selected_book_tuple[1]
        book_details = df_combined_books[df_combined_books['book_id'] == selected_book_id]
        if not book_details.empty:
            display_book_preview(book_details)
        else:
            st.warning("Detalles del libro no encontrados.")

    rating = st.slider('Calificación (1-5):', 1, 5, 3, key='book_rating_slider')

    if st.button('Añadir Calificación', key='add_rating_button'):
        if selected_book_tuple and selected_book_tuple[1] is not None:
            book_id_to_add = selected_book_tuple[1]
            # Evitar añadir el mismo libro dos veces por el mismo usuario
            if book_id_to_add not in [r['book_id'] for r in st.session_state.new_user_ratings_collected]:
                st.session_state.new_user_ratings_collected.append({
                    'user_id': st.session_state.new_user_id,
                    'book_id': book_id_to_add,
                    'rating': rating
                })
                st.success(f"Se añadió la calificación de {rating} para '{selected_book_tuple[0]}'.")
                # Actualizar el conjunto de libros calificados por el usuario objetivo
                st.session_state.target_user_rated_books_set.add(book_id_to_add)
            else:
                st.warning("Ya has calificado este libro. Edita la calificación existente si lo deseas.")
        else:
            st.warning("Por favor, selecciona un libro antes de calificar.")

    st.subheader("Tus Calificaciones:")
    if st.session_state.new_user_ratings_collected:
        # Crea un DataFrame temporal para mostrar las calificaciones
        rated_books_df_display = pd.DataFrame(st.session_state.new_user_ratings_collected)
        
        # Unir con df_combined_books para obtener detalles del libro
        # CORRECCIÓN: Asegúrate de que 'authors' y no 'author' sea la columna que esperas
        rated_books_display = rated_books_df_display.merge(
            df_combined_books[['book_id', 'title', 'authors', 'image_url']], on='book_id', how='left'
        )
        
        # Limpiar los títulos de los libros mostrados en "Tus Calificaciones"
        rated_books_display['title_clean'] = rated_books_display['title'].apply(lambda x: re.sub(r'\s*\(.*?\)\s*', '', str(x)).strip())
        
        # Ajusta cómo se muestra el autor. Usa 'authors' o 'author' dependiendo de lo que tengas.
        # Aquí asumo que df_combined_books tiene una columna 'authors'.
        # Si tu df_combined_books tiene 'author' en singular, ajusta esto.
        # Si df_combined_books tiene solo 'authors' (plural), puedes usar solo eso.
        rated_books_display['Author_Display'] = rated_books_display.apply(
            lambda row: row.get('authors', 'Desconocido'), axis=1
        )
        
        # Muestra en formato de tabla para claridad
        for index, row in rated_books_display.iterrows():
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 5px; padding: 5px; border: 1px solid #eee; border-radius: 5px; background-color: #fcfcfc;">
                <img src="{row.get('image_url', 'https://via.placeholder.com/50x70?text=No+Cover')}" width="50" height="70" style="margin-right: 10px; border-radius: 3px;">
                <div>
                    <strong>{row['title_clean']}</strong> por {row['Author_Display']} <br>
                    Calificación: {row['rating']} ⭐
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Botón para borrar calificaciones
        if st.button("Borrar todas mis calificaciones", key='clear_ratings_button'):
            st.session_state.new_user_ratings_collected = []
            st.session_state.recommendations_generated = False
            st.session_state.target_user_rated_books_set = set()
            st.session_state.recommended_book_ids_set = set()
            st.session_state.new_user_data['reading_preference'] = 'Desconocida' # Reiniciar preferencia
            st.success("Todas las calificaciones han sido borradas.")
            st.rerun()
    else:
        st.info("Aún no has calificado ningún libro.")

# --- Lógica de generación de recomendaciones ---
st.markdown('<h2 class="section-header">Tus Recomendaciones Personalizadas</h2>', unsafe_allow_html=True)

if st.button('Generar Recomendaciones', key='generate_recs_button'):
    if len(st.session_state.new_user_ratings_collected) < 3:
        st.warning("Por favor, califica al menos 3 libros para obtener recomendaciones.")
    else:
        with st.spinner("Generando tus recomendaciones... esto puede tardar un momento."):
            # 1. Preparar los datos de calificación del nuevo usuario
            new_user_df = pd.DataFrame(st.session_state.new_user_ratings_collected)
            
            # Asignar preferencia de lectura basada en las calificaciones actuales
            st.session_state.new_user_data['reading_preference'] = assign_reading_preference(
                new_user_df, df_combined_books, unique_genres
            )

            # 2. Combinar calificaciones históricas con las del nuevo usuario
            # Asegúrate de que las columnas coincidan ('user_id', 'book_id', 'rating')
            combined_ratings_df = pd.concat([df_ratings_modified, new_user_df], ignore_index=True)

            # 3. Re-entrenar el modelo SVD con los datos combinados
            reader = Reader(rating_scale=(1, 5))
            combined_data = Dataset.load_from_df(combined_ratings_df[['user_id', 'book_id', 'rating']], reader)
            trainset = combined_data.build_full_trainset()
            
            # Si ya hay un modelo dinámico, podrías intentar un "warm-start" o simplemente re-entrenar desde cero
            # Para simplicidad y robustez, re-entrenaremos desde cero aquí.
            st.session_state.dynamic_svd_algo = SVD(n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
            st.session_state.dynamic_svd_algo.fit(trainset)
            
            # 4. Generar predicciones para el nuevo usuario
            # Obtener todos los book_ids únicos
            all_book_ids = df_combined_books['book_id'].unique()
            
            # Excluir los libros que el usuario ya ha calificado
            # Convertir a conjunto para una búsqueda eficiente
            st.session_state.target_user_rated_books_set = set(new_user_df['book_id'].unique())
            books_to_predict = [book_id for book_id in all_book_ids if book_id not in st.session_state.target_user_rated_books_set]

            predictions = []
            for book_id in books_to_predict:
                pred = st.session_state.dynamic_svd_algo.predict(st.session_state.new_user_id, book_id)
                predictions.append((book_id, pred.est))
            
            # Ordenar las predicciones por calificación estimada descendente
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Obtener los 10-20 libros mejor calificados para recomendar
            num_recommendations = 20
            st.session_state.recommendations = predictions[:num_recommendations]
            
            # Almacenar los IDs de los libros recomendados para evitar duplicados en otras secciones
            st.session_state.recommended_book_ids_set = {book_id for book_id, _ in st.session_state.recommendations}

            # 5. Encontrar usuarios similares usando KNN
            # Añadir el nuevo usuario a user_item_matrix para encontrar similares
            # Es necesario reconstruir user_item_matrix con el nuevo usuario
            temp_user_item_matrix = combined_ratings_df.pivot_table(
                index='user_id', columns='book_id', values='rating'
            ).fillna(0) # Rellenar NaN con 0 para la matriz
            
            # Asegurarse de que las columnas de la matriz coincidan con df_combined_books
            # Añadir columnas de libros que puedan faltar en temp_user_item_matrix
            missing_cols = set(df_combined_books['book_id'].unique()) - set(temp_user_item_matrix.columns)
            for col in missing_cols:
                temp_user_item_matrix[col] = 0
            temp_user_item_matrix = temp_user_item_matrix[df_combined_books['book_id'].unique()] # Ordenar columnas
            
            # Recalcular KNN sobre esta matriz temporal actualizada
            # Asegurarse de que el nuevo usuario esté en el índice para obtener los vecinos
            if st.session_state.new_user_id in temp_user_item_matrix.index:
                query_user_data = temp_user_item_matrix.loc[st.session_state.new_user_id].values.reshape(1, -1)
                
                # Excluir al propio usuario de la búsqueda de vecinos si user_item_matrix incluye solo usuarios antiguos
                # Si temp_user_item_matrix ya incluye al nuevo usuario, es importante no encontrarse a sí mismo.
                # Depende de cómo se construye el trainset de KNN.
                # Asumiendo que knn_best ya fue entrenado con user_item_matrix original.
                # Aquí, queremos encontrar vecinos para el nuevo usuario.
                # La forma más segura es re-entrenar knn o asegurar que el nuevo usuario no sea su propio vecino más cercano.
                # Para simplificar, asumiremos que KNN buscará vecinos en la matriz original y el nuevo usuario se compara con ellos.
                
                # KNN devuelve distancias y índices
                distances, indices = knn_best.kneighbors(query_user_data)
                
                # Filtrar distancias e índices inválidos (ej. si knn_best no encontró vecinos o si hay 0s)
                # distances y indices son arrays 2D porque query_user_data es 2D
                valid_indices = indices[0][distances[0] > 0] # Excluir distancia 0 (propio usuario)
                valid_distances = distances[0][distances[0] > 0]
                
                # Obtener los user_id de los vecinos más cercanos
                # Los índices de KNN corresponden a las filas de la matriz con la que se entrenó
                # Necesitamos mapear esos índices de vuelta a los user_id originales
                
                # Asumiendo que user_item_matrix.index tiene los user_id correctos en el orden de entrenamiento de knn_best
                similar_user_ids_mapped = [user_item_matrix.index[idx] for idx in valid_indices]
                
                st.session_state.similar_users_info = []
                for i, sim_user_id in enumerate(similar_user_ids_mapped):
                    sim_user_data = df_users[df_users['user_id'] == sim_user_id].iloc[0]
                    st.session_state.similar_users_info.append({
                        'user_id': sim_user_id,
                        'username': sim_user_data.get('username', f'Usuario {sim_user_id}'),
                        'avatar_url': sim_user_data.get('avatar_url', f"https://api.dicebear.com/7.x/identicon/svg?seed={sim_user_id}"),
                        'similarity': 1 - valid_distances[i] # Convertir distancia a similitud (ej. 1 - distancia)
                    })
                st.session_state.similar_users_info = pd.DataFrame(st.session_state.similar_users_info)
                st.session_state.similar_users_info = st.session_state.similar_users_info.sort_values(by='similarity', ascending=False)
            else:
                st.warning("No se pudo encontrar al nuevo usuario en la matriz de ítems para KNN.")
                st.session_state.similar_users_info = pd.DataFrame()

            st.session_state.recommendations_generated = True
            st.success("¡Recomendaciones generadas con éxito!")
            st.rerun() 

if st.session_state.recommendations_generated:
    recommended_books_details = []
    for book_id, estimated_score in st.session_state.recommendations:
        details = df_combined_books[df_combined_books['book_id'] == int(book_id)]
        if not details.empty:
            recommended_books_details.append((book_id, estimated_score, details.iloc[0]))
    
    display_books_grid(recommended_books_details, "Libros Recomendados para Ti")

    # Sección de Usuarios Similares
    st.markdown('<h3 class="section-header">Explora los gustos de usuarios similares</h3>', unsafe_allow_html=True)
    if not st.session_state.similar_users_info.empty:
        # Usamos una scrollbar horizontal para los avatares
        st.markdown('<div class="horizontal-scroll">', unsafe_allow_html=True)
        cols_similar = st.columns(len(st.session_state.similar_users_info))
        
        for i, (idx, row) in enumerate(st.session_state.similar_users_info.iterrows()):
            with cols_similar[i]:
                if st.button(key=f"sim_user_{row['user_id']}_button", label="", help=f"Ver libros de {row['username']}"):
                    st.session_state.selected_similar_user_id = row['user_id']
                
                # Muestra la tarjeta del avatar y el nombre/similitud
                st.markdown(f"""
                <div class="avatar-card">
                    <img src="{row['avatar_url']}" alt="{row['username']}" width="80" height="80">
                    <p class="username">{row['username']}</p>
                    <p class="similarity">Similitud: {row['similarity']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Mostrar libros del usuario similar seleccionado
        if 'selected_similar_user_id' in st.session_state and st.session_state.selected_similar_user_id is not None:
            st.markdown(f'<h4 style="margin-top: 20px;">Libros calificados por {st.session_state.similar_users_info[st.session_state.similar_users_info["user_id"] == st.session_state.selected_similar_user_id]["username"].iloc[0]}:</h4>', unsafe_allow_html=True)
            if st.session_state.dynamic_svd_algo is not None:
                display_similar_user_rated_books(
                    st.session_state.selected_similar_user_id,
                    st.session_state.target_user_rated_books_set,
                    st.session_state.recommended_book_ids_set, # Pasar los libros ya recomendados también
                    df_combined_books,
                    user_item_matrix, # Usar la matriz original para el lookup de ratings históricos
                    st.session_state.dynamic_svd_algo,
                    st.session_state.new_user_id
                )
            else:
                st.warning("El modelo SVD no está listo para predecir libros de usuarios similares.")
    else:
        st.info("No se encontraron usuarios similares. Califica más libros para obtener mejores resultados.")