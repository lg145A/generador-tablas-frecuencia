import math
import re
import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib
matplotlib.use('Agg') # Necesario para generar imágenes sin interfaz gráfica
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pypdf
import pytesseract
from fpdf import FPDF

# --- CONFIGURACIÓN DE TESSERACT (Para leer imágenes) ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
st.set_page_config(
    page_title="Generador de Tablas de Frecuencia",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #3498db, #2c3e50);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title { font-size: 2.5rem; margin: 0; font-weight: bold; }
    .header-subtitle { font-size: 1.2rem; opacity: 0.9; margin-top: 0.5rem; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 5px solid #3498db;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #3498db; margin-bottom: 0.5rem; }
    .metric-label { font-size: 0.9rem; color: #34495e; text-transform: uppercase; letter-spacing: 1px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: white; border-radius: 5px 5px 0 0;
        font-weight: bold; color: #2c3e50; box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] { background-color: #3498db; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNCIONES ESTADÍSTICAS ---
def parse_data(texto: str) -> list[float]:
    texto = str(texto).replace(";", " ").replace("\n", " ").replace("\t", " ")
    texto = re.sub(r",\s+", " ", texto)
    texto = re.sub(r",\s*$", "", texto)
    partes = texto.split()
    if not partes: return []
    if len(partes) == 1 and "," in partes[0]:
        partes = [p for p in partes[0].split(",") if p.strip() != ""]
    valores = []
    for p in partes:
        p = p.strip()
        if p == "": continue
        if "," in p and "." not in p:
            if p.count(",") == 1:
                p = p.replace(",", ".")
            else:
                for sub in p.split(","):
                    if sub.strip():
                        valores.append(float(sub.strip()))
                continue
        try:
            v = float(p)
            if math.isfinite(v): valores.append(v)
        except ValueError:
            continue
    return valores

def regla_sturges(n: int) -> int:
    if n <= 1: return 1
    return max(1, math.ceil(1 + 3.322 * math.log10(n)))

def construir_tabla(datos: list[float]):
    arr = np.asarray(datos, dtype=float)
    n = int(arr.size)
    if n == 0: raise ValueError("No hay datos.")

    minimo_dato = float(np.min(arr))
    maximo_dato = float(np.max(arr))
    
    minimo = math.floor(minimo_dato)
    maximo = math.ceil(maximo_dato)
    rango = maximo - minimo

    if rango == 0:
        k, amplitud = 1, 1
        intervalos = [f"[{minimo}, {maximo}]"]
        marcas = [(minimo + maximo) / 2]
        fi = np.array([n], dtype=int)
        edges = np.array([minimo, maximo], dtype=float)
    else:
        k = regla_sturges(n)
        amplitud = math.ceil(rango / k)
        
        edges = [minimo + i * amplitud for i in range(k + 1)]
        if edges[-1] < maximo_dato:
            edges[-1] = math.ceil(maximo_dato)

        fi = np.zeros(k, dtype=int)

        for x in arr:
            idx = k - 1
            for i in range(k):
                if x >= edges[i] and x < edges[i+1]:
                    idx = i
                    break
            fi[idx] += 1

        intervalos, marcas = [], []
        for i in range(k):
            li, ls = float(edges[i]), float(edges[i + 1])
            if i == k - 1:
                intervalos.append(f"[{li:.0f}, {ls:.0f}]")
            else:
                intervalos.append(f"[{li:.0f}, {ls:.0f})")
            marcas.append((li + ls) / 2)

    Fi = np.cumsum(fi)
    hi = fi / n
    porcentaje = hi * 100

    df = pd.DataFrame({
        "Intervalo de clase": intervalos, 
        "Marca de clase (Xi)": marcas, 
        "Frecuencia absoluta (fi)": fi, 
        "Frecuencia acumulada (Fi)": Fi, 
        "Frecuencia relativa (hi)": hi, 
        "Frecuencia porcentual (%)": porcentaje
    })

    info = {"n": n, "minimo": minimo_dato, "maximo": maximo_dato, "rango": maximo_dato - minimo_dato, "k": k, "amplitud": amplitud}
    return df, info

# --- 3. GENERACIÓN DE GRÁFICOS ESTÁTICOS (Para PDF, Excel y PNG) ---
def generar_graficos_matplotlib(df, info, datos_raw):
    colores = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C7C7C7', '#5366FF', '#289F40', '#D26384']
    
    # 1. Gráfico de Barras
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    ax_bar.bar(df["Intervalo de clase"], df["Frecuencia absoluta (fi)"], color=colores[:len(df)], edgecolor='black')
    ax_bar.set_title("Gráfico de Barras")
    ax_bar.tick_params(axis='x', rotation=45)
    buf_bar = io.BytesIO()
    fig_bar.savefig(buf_bar, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig_bar)
    buf_bar.seek(0)

    # 2. Gráfico Circular
    fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
    ax_pie.pie(df["Frecuencia absoluta (fi)"], labels=df["Intervalo de clase"], autopct='%1.1f%%', colors=colores[:len(df)])
    ax_pie.set_title("Gráfico Circular")
    buf_pie = io.BytesIO()
    fig_pie.savefig(buf_pie, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig_pie)
    buf_pie.seek(0)

    # 3. Histograma
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    if datos_raw:
        ax_hist.hist(datos_raw, bins=info['k'], color='#72B7B2', edgecolor='black')
    ax_hist.set_title("Histograma")
    buf_hist = io.BytesIO()
    fig_hist.savefig(buf_hist, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig_hist)
    buf_hist.seek(0)

    return buf_bar, buf_pie, buf_hist

# --- 4. FUNCIONES DE EXPORTACIÓN ---
def preparar_tabla_exportacion(df):
    df_exp = df.copy()
    df_exp["Marca de clase (Xi)"] = df_exp["Marca de clase (Xi)"].map(lambda x: f"{x:.2f}")
    df_exp["Frecuencia relativa (hi)"] = df_exp["Frecuencia relativa (hi)"].map(lambda x: f"{x:.4f}")
    df_exp["Frecuencia porcentual (%)"] = df_exp["Frecuencia porcentual (%)"].map(lambda x: f"{x:.2f}%")
    return df_exp

def generar_excel_tabla(df_fmt, info, nombre_tabla, datos_raw):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_fmt.to_excel(writer, sheet_name='Reporte', startrow=10, index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Reporte']
        
        title_format = workbook.add_format({'bold': True, 'font_size': 14})
        label_format = workbook.add_format({'bold': True})
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center',
            'fg_color': '#3498db', 'font_color': 'white', 'border': 1
        })
        
        worksheet.write(0, 0, "TABLA DE DISTRIBUCIÓN DE FRECUENCIAS", title_format)
        
        fecha_actual = datetime.now().strftime("%d/%m/%Y")
        metadata = [
            ("Nombre:", nombre_tabla), ("Fecha:", fecha_actual),
            ("Total de datos (n):", info['n']), ("Valor mínimo:", f"{info['minimo']:.2f}"),
            ("Valor máximo:", f"{info['maximo']:.2f}"), ("Rango:", f"{info['rango']:.2f}"),
            ("Número de clases (k):", info['k']), ("Amplitud de clase:", f"{info['amplitud']:.0f}")
        ]
        
        for i, (label, value) in enumerate(metadata):
            worksheet.write(2 + i, 0, label, label_format)
            worksheet.write(2 + i, 1, value)
            
        for col_num, value in enumerate(df_fmt.columns.values):
            worksheet.write(10, col_num, value, header_format)
            
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:F', 25)
        
        # Insertar Gráficos en Excel
        buf_bar, buf_pie, buf_hist = generar_graficos_matplotlib(df_fmt, info, datos_raw)
        worksheet.insert_image(22, 0, "bar.png", {'image_data': buf_bar})
        worksheet.insert_image(22, 3, "pie.png", {'image_data': buf_pie})
        worksheet.insert_image(45, 0, "hist.png", {'image_data': buf_hist})
        
    return buffer.getvalue()

def generar_pdf_tabla(df_fmt, info, nombre_tabla, datos_raw):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    # --- PÁGINA 1: TABLA ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="TABLA DE DISTRIBUCIÓN DE FRECUENCIAS", ln=True, align='L')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=11)
    fecha_actual = datetime.now().strftime("%d/%m/%Y")
    pdf.cell(0, 7, txt=f"Nombre: {nombre_tabla}   |   Fecha: {fecha_actual}", ln=True)
    pdf.cell(0, 7, txt=f"Total datos: {info['n']}   |   Mín: {info['minimo']:.2f}   |   Máx: {info['maximo']:.2f}   |   Rango: {info['rango']:.2f}", ln=True)
    pdf.cell(0, 7, txt=f"N° clases: {info['k']}   |   Amplitud: {info['amplitud']:.0f}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(52, 152, 219)
    pdf.set_text_color(255, 255, 255)
    
    cols = df_fmt.columns
    w = 260 / len(cols)
    
    for col in cols:
        pdf.cell(w, 8, txt=str(col), border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0)
    
    for i in range(len(df_fmt)):
        if i % 2 == 0: pdf.set_fill_color(255, 255, 255)
        else: pdf.set_fill_color(242, 242, 242)
        for col in cols:
            pdf.cell(w, 8, txt=str(df_fmt.iloc[i][col]), border=1, align='C', fill=True)
        pdf.ln()

    # --- PÁGINA 2: GRÁFICOS ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="REPRESENTACIÓN GRÁFICA", ln=True, align='C')
    pdf.ln(5)
    
    buf_bar, buf_pie, buf_hist = generar_graficos_matplotlib(df_fmt, info, datos_raw)
    
    # Insertar imágenes (X, Y, Ancho)
    pdf.image(buf_bar, x=10, y=30, w=130)
    pdf.image(buf_pie, x=150, y=30, w=110)
    pdf.image(buf_hist, x=10, y=110, w=130)
    
    return bytes(pdf.output())

def generar_png_tabla(df_fmt, info, nombre_tabla, datos_raw):
    fig = plt.figure(figsize=(16, 18))
    
    # Título y Metadatos
    fecha_actual = datetime.now().strftime("%d/%m/%Y")
    fig.text(0.05, 0.95, "TABLA DE DISTRIBUCIÓN DE FRECUENCIAS", fontsize=20, weight='bold')
    metadatos = (
        f"Nombre: {nombre_tabla}   |   Fecha: {fecha_actual}\n"
        f"Total de datos: {info['n']}   |   Mín: {info['minimo']:.2f}   |   Máx: {info['maximo']:.2f}\n"
        f"Rango: {info['rango']:.2f}   |   N° clases: {info['k']}   |   Amplitud: {info['amplitud']:.0f}"
    )
    fig.text(0.05, 0.90, metadatos, fontsize=12, linespacing=1.6)
    
    # Tabla
    ax_table = fig.add_axes([0.05, 0.55, 0.9, 0.3])
    ax_table.axis('off')
    tabla = ax_table.table(cellText=df_fmt.values, colLabels=df_fmt.columns, cellLoc='center', loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    for (i, j), cell in tabla.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#3498db')
        elif i % 2 == 0:
            cell.set_facecolor('#f2f2f2')
            
    colores = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C7C7C7', '#5366FF', '#289F40', '#D26384']
    
    # Gráfico de Barras
    ax_bar = fig.add_axes([0.05, 0.28, 0.4, 0.2])
    ax_bar.bar(df_fmt["Intervalo de clase"], df_fmt["Frecuencia absoluta (fi)"], color=colores[:len(df_fmt)], edgecolor='black')
    ax_bar.set_title("Gráfico de Barras")
    ax_bar.tick_params(axis='x', rotation=45)
    
    # Gráfico Circular
    ax_pie = fig.add_axes([0.55, 0.28, 0.4, 0.2])
    ax_pie.pie(df_fmt["Frecuencia absoluta (fi)"], labels=df_fmt["Intervalo de clase"], autopct='%1.1f%%', colors=colores[:len(df_fmt)])
    ax_pie.set_title("Gráfico Circular")
    
    # Histograma
    ax_hist = fig.add_axes([0.25, 0.03, 0.5, 0.2])
    if datos_raw:
        ax_hist.hist(datos_raw, bins=info['k'], color='#72B7B2', edgecolor='black')
    ax_hist.set_title("Histograma")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --- FUNCIÓN PARA PROCESAR TEXTO (PDF o Imagen) ---
def procesar_texto_extraido(texto_extraido):
    texto_limpio = texto_extraido.replace('|', ' ').replace('\n', ' ')
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
    
    patron_tabla = r"\[?\s*([\d\.]+)\s*[,.]\s*([\d\.]+)\s*[\]\)]?\s+([\d\.]+)\s+(\d+)\s+(\d+)\s+([\d\.]+)"
    coincidencias = re.findall(patron_tabla, texto_limpio)

    if coincidencias:
        intervalos, marcas, fi_list, Fi_list, hi_list, porcentaje_list = [], [], [], [], [], []
        for m in coincidencias:
            intervalos.append(f"[{m[0]}, {m[1]})") 
            marcas.append(float(m[2]))
            fi_list.append(int(m[3]))
            Fi_list.append(int(m[4]))
            hi_list.append(float(m[5]))
            porcentaje_list.append(float(m[5]) * 100)

        if intervalos:
             intervalos[-1] = intervalos[-1].replace(")", "]")

        df_temp = pd.DataFrame({
            "Intervalo de clase": intervalos, "Marca de clase (Xi)": marcas,
            "Frecuencia absoluta (fi)": fi_list, "Frecuencia acumulada (Fi)": Fi_list, 
            "Frecuencia relativa (hi)": hi_list, "Frecuencia porcentual (%)": porcentaje_list
        })

        datos_sinteticos = []
        for i, intervalo in enumerate(intervalos):
            nums = re.findall(r"[\d\.]+", intervalo)
            if len(nums) >= 2:
                li, ls = float(nums[0]), float(nums[1])
                if fi_list[i] > 0:
                    datos_sinteticos.extend(np.linspace(li, ls, fi_list[i], endpoint=True).tolist())

        n = sum(fi_list)
        min_val = float(re.findall(r"[\d\.]+", intervalos[0])[0])
        max_val = float(re.findall(r"[\d\.]+", intervalos[-1])[1])
        k = len(intervalos)
        info = {"n": n, "minimo": min_val, "maximo": max_val, "rango": max_val - min_val, "k": k, "amplitud": (max_val - min_val)/k if k > 0 else 0}

        return df_temp, datos_sinteticos, info, ", ".join([f"{x:.2f}" for x in datos_sinteticos]), "✅ Tabla de frecuencias detectada y cargada exactamente."
    
    numeros = parse_data(texto_extraido)
    if len(numeros) >= 5:
        return None, None, None, ", ".join([str(x) for x in numeros]), f"✅ Se extrajeron {len(numeros)} datos crudos."
    else:
        return None, None, None, None, "No se encontraron suficientes números."

# --- 5. GESTIÓN DE ESTADO ---
ARCHIVO_TABLAS = "tablas_guardadas.json"
if 'texto_input' not in st.session_state: st.session_state.texto_input = ""
if 'datos_procesados' not in st.session_state: st.session_state.datos_procesados = None
if 'df' not in st.session_state: st.session_state.df = None
if 'info' not in st.session_state: st.session_state.info = None
if 'nombre_actual' not in st.session_state: st.session_state.nombre_actual = ""
if 'msj_error_archivo' not in st.session_state: st.session_state.msj_error_archivo = ""
if 'msj_exito_archivo' not in st.session_state: st.session_state.msj_exito_archivo = ""

if os.path.exists(ARCHIVO_TABLAS):
    with open(ARCHIVO_TABLAS, "r", encoding="utf-8") as f:
        st.session_state.tablas_guardadas = json.load(f)
else:
    st.session_state.tablas_guardadas = []

def guardar_tablas_en_disco():
    with open(ARCHIVO_TABLAS, "w", encoding="utf-8") as f:
        json.dump(st.session_state.tablas_guardadas, f, ensure_ascii=False, indent=4)

def generar_aleatorios():
    st.session_state.texto_input = ", ".join([str(np.random.randint(10, 100)) for _ in range(50)])
    st.session_state.datos_procesados = None; st.session_state.df = None; st.session_state.info = None
    st.session_state.msj_error_archivo = ""; st.session_state.msj_exito_archivo = ""

def limpiar_todo():
    st.session_state.texto_input = ""
    st.session_state.datos_procesados = None; st.session_state.df = None; st.session_state.info = None
    st.session_state.msj_error_archivo = ""; st.session_state.msj_exito_archivo = ""
    if "nombre_tabla_input" in st.session_state: st.session_state.nombre_tabla_input = ""

def handle_file_upload():
    archivo = st.session_state.get("file_uploader_key")
    st.session_state.msj_error_archivo = ""
    st.session_state.msj_exito_archivo = ""
    
    if archivo is not None:
        try:
            ext = archivo.name.split('.')[-1].lower()
            st.session_state.nombre_actual = f"Tabla - {archivo.name}"
            
            if ext in ['jpg', 'jpeg', 'png']:
                imagen = Image.open(archivo)
                imagen = ImageOps.grayscale(imagen)
                imagen = imagen.point(lambda x: 0 if x < 150 else 255, '1') 
                texto_extraido = pytesseract.image_to_string(imagen, config='--psm 6')
                df_temp, datos_sinteticos, info, texto_input, msj = procesar_texto_extraido(texto_extraido)
                
                if df_temp is not None:
                    st.session_state.df, st.session_state.datos_procesados, st.session_state.info = df_temp, datos_sinteticos, info
                    st.session_state.texto_input = texto_input
                    st.session_state.msj_exito_archivo = msj
                elif texto_input is not None:
                    st.session_state.texto_input = texto_input; st.session_state.df = None
                    st.session_state.msj_exito_archivo = msj
                else:
                    st.session_state.msj_error_archivo = msj
                return

            elif ext == 'pdf':
                reader = pypdf.PdfReader(archivo)
                texto_extraido = "".join([page.extract_text() for page in reader.pages])
                df_temp, datos_sinteticos, info, texto_input, msj = procesar_texto_extraido(texto_extraido)
                
                if df_temp is not None:
                    st.session_state.df, st.session_state.datos_procesados, st.session_state.info = df_temp, datos_sinteticos, info
                    st.session_state.texto_input = texto_input
                    st.session_state.msj_exito_archivo = msj
                elif texto_input is not None:
                    st.session_state.texto_input = texto_input; st.session_state.df = None
                    st.session_state.msj_exito_archivo = msj
                else:
                    st.session_state.msj_error_archivo = msj
                return

            elif ext in ['csv', 'xlsx', 'xls']:
                if ext == 'csv':
                    df_temp = pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8-sig')
                else:
                    raw_excel = pd.read_excel(archivo, header=None)
                    header_row = None
                    for idx, row in raw_excel.iterrows():
                        row_str = [str(cell).strip() for cell in row.values]
                        if any("Intervalo" in cell for cell in row_str) and any("Marca" in cell or "fi" in cell.lower() for cell in row_str):
                            header_row = idx
                            break
                    
                    if header_row is not None:
                        df_temp = pd.read_excel(archivo, header=header_row)
                        rename_dict = {}
                        for col in df_temp.columns:
                            col_str = str(col).strip()
                            if "Intervalo" in col_str: rename_dict[col] = "Intervalo de clase"
                            elif "Marca" in col_str: rename_dict[col] = "Marca de clase (Xi)"
                            elif "Abs" in col_str or col_str == "fi": rename_dict[col] = "Frecuencia absoluta (fi)"
                            elif "Acum" in col_str or col_str == "Fi": rename_dict[col] = "Frecuencia acumulada (Fi)"
                            elif "Rel" in col_str or col_str == "hi": rename_dict[col] = "Frecuencia relativa (hi)"
                            elif "%" in col_str: rename_dict[col] = "Frecuencia porcentual (%)"
                        df_temp.rename(columns=rename_dict, inplace=True)
                    else:
                        df_temp = pd.read_excel(archivo)
                
                columnas_esperadas = ["Intervalo de clase", "Frecuencia absoluta (fi)", "Frecuencia acumulada (Fi)", "Marca de clase (Xi)"]
                
                if not all(col in df_temp.columns for col in columnas_esperadas):
                    for col in df_temp.columns:
                        c_lower = str(col).lower()
                        if "intervalo" in c_lower: df_temp.rename(columns={col: "Intervalo de clase"}, inplace=True)
                        elif "marca" in c_lower: df_temp.rename(columns={col: "Marca de clase (Xi)"}, inplace=True)
                        elif "abs" in c_lower or c_lower == "fi": df_temp.rename(columns={col: "Frecuencia absoluta (fi)"}, inplace=True)
                        elif "acum" in c_lower or c_lower == "fi": df_temp.rename(columns={col: "Frecuencia acumulada (Fi)"}, inplace=True)
                        elif "rel" in c_lower or c_lower == "hi": df_temp.rename(columns={col: "Frecuencia relativa (hi)"}, inplace=True)
                        elif "%" in c_lower: df_temp.rename(columns={col: "Frecuencia porcentual (%)"}, inplace=True)

                if all(col in df_temp.columns for col in columnas_esperadas):
                    if "Frecuencia porcentual (%)" in df_temp.columns:
                        df_temp["Frecuencia porcentual (%)"] = df_temp["Frecuencia porcentual (%)"].astype(str).str.replace("%", "").astype(float)
                    
                    df_temp["Frecuencia absoluta (fi)"] = df_temp["Frecuencia absoluta (fi)"].astype(int)
                    df_temp["Frecuencia acumulada (Fi)"] = df_temp["Frecuencia acumulada (Fi)"].astype(int)
                    df_temp["Marca de clase (Xi)"] = df_temp["Marca de clase (Xi)"].astype(float)
                    df_temp["Frecuencia relativa (hi)"] = df_temp["Frecuencia relativa (hi)"].astype(float)
                    
                    if "Frecuencia porcentual (%)" not in df_temp.columns:
                        df_temp["Frecuencia porcentual (%)"] = df_temp["Frecuencia relativa (hi)"] * 100
                        
                    df_temp = df_temp[["Intervalo de clase", "Marca de clase (Xi)", "Frecuencia absoluta (fi)", "Frecuencia acumulada (Fi)", "Frecuencia relativa (hi)", "Frecuencia porcentual (%)"]]
                    st.session_state.df = df_temp
                    
                    datos_sinteticos = []
                    for _, row in df_temp.iterrows():
                        nums = re.findall(r"[\d\.]+", str(row["Intervalo de clase"]))
                        if len(nums) >= 2:
                            li, ls = float(nums[0]), float(nums[1])
                            if int(row["Frecuencia absoluta (fi)"]) > 0:
                                datos_sinteticos.extend(np.linspace(li, ls, int(row["Frecuencia absoluta (fi)"]), endpoint=True).tolist())
                    
                    st.session_state.datos_procesados = datos_sinteticos
                    
                    nums_first = re.findall(r"[\d\.]+", str(df_temp["Intervalo de clase"].iloc[0]))
                    nums_last = re.findall(r"[\d\.]+", str(df_temp["Intervalo de clase"].iloc[-1]))
                    
                    min_val = float(nums_first[0]) if len(nums_first) >= 1 else 0
                    max_val = float(nums_last[1]) if len(nums_last) >= 2 else (float(nums_last[0]) if len(nums_last) == 1 else 0)
                    
                    n = sum(df_temp["Frecuencia absoluta (fi)"])
                    k = len(df_temp)
                    st.session_state.info = {"n": n, "minimo": min_val, "maximo": max_val, "rango": max_val - min_val, "k": k, "amplitud": (max_val - min_val)/k if k > 0 else 0}
                    st.session_state.texto_input = ", ".join([f"{x:.2f}" for x in datos_sinteticos])
                    st.session_state.msj_exito_archivo = f"✅ Tabla de frecuencias cargada exactamente desde {archivo.name}."
                else:
                    numeros = df_temp.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values.flatten().tolist()
                    if len(numeros) >= 5:
                        st.session_state.texto_input = ", ".join([str(x) for x in numeros])
                        st.session_state.df = None
                        st.session_state.msj_exito_archivo = f"✅ Se cargaron {len(numeros)} datos crudos."
        except Exception as e:
            st.session_state.msj_error_archivo = f"Error al procesar {archivo.name}: {e}"
            st.session_state.df = None
            st.session_state.info = None
            st.session_state.datos_procesados = None

# --- 6. INTERFAZ PRINCIPAL ---
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Generador de Tablas de Frecuencia</h1>
    <p class="header-subtitle">Aplicación estadística interactiva</p>
</div>
""", unsafe_allow_html=True)

tab_input, tab_results, tab_charts, tab_saved = st.tabs([
    "📥 Ingreso de Datos", "📋 Resultados", "📈 Gráficos", "💾 Tablas Guardadas"
])

# --- PESTAÑA 1: INGRESO DE DATOS ---
with tab_input:
    st.info("**Instrucciones:** Ingresa datos manualmente o carga un archivo (CSV, Excel, PDF, Imagen).")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.text_input("Nombre de la Tabla (Opcional):", placeholder="Ej: Análisis de Calificaciones", key="nombre_tabla_input")
        col_btn_rand, col_btn_clear = st.columns(2)
        with col_btn_rand:
            st.button("🎲 Generar Datos Aleatorios", on_click=generar_aleatorios, use_container_width=True)
        with col_btn_clear:
            st.button("🗑️ Limpiar Todo", on_click=limpiar_todo, use_container_width=True)
        st.text_area("Ingresar Datos Manualmente:", height=150, key="texto_input")

    with col2:
        st.write("Cargar desde archivo:")
        st.file_uploader(
            "Sube un archivo", 
            type=['csv', 'xlsx', 'xls', 'pdf', 'jpg', 'jpeg', 'png'], 
            key="file_uploader_key", 
            on_change=handle_file_upload
        )
        if st.session_state.msj_exito_archivo: st.success(st.session_state.msj_exito_archivo)
        if st.session_state.msj_error_archivo: st.error(st.session_state.msj_error_archivo)

    st.write("---")
    if st.button("🧮 Procesar Datos", use_container_width=True, type="primary"):
        try:
            datos = parse_data(st.session_state.texto_input)
            if len(datos) < 5:
                st.warning("Se necesitan al menos 5 datos.")
            else:
                st.session_state.datos_procesados = datos
                st.session_state.df, st.session_state.info = construir_tabla(datos)
                nombre = st.session_state.get("nombre_tabla_input", "")
                st.session_state.nombre_actual = nombre if nombre else f"Tabla - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
                st.success("Datos procesados correctamente. Revisa 'Resultados'.")
                st.session_state.msj_exito_archivo = ""
                st.session_state.msj_error_archivo = ""
        except Exception as e:
            st.error(f"Error: {e}")

# --- PESTAÑA 2: RESULTADOS ---
with tab_results:
    if st.session_state.df is not None and st.session_state.info is not None:
        info = st.session_state.info
        df = st.session_state.df
        datos_raw = st.session_state.datos_procesados

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{info["n"]}</div><div class="metric-label">Total Datos</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{info["minimo"]:.2f}</div><div class="metric-label">Mínimo</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{info["maximo"]:.2f}</div><div class="metric-label">Máximo</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-value">{info["k"]}</div><div class="metric-label">N° Clases</div></div>', unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("Tabla de Distribución de Frecuencias")
        
        df_fmt_view = df.copy()
        df_fmt_view["Marca de clase (Xi)"] = df_fmt_view["Marca de clase (Xi)"].map(lambda x: f"{x:.2f}")
        df_fmt_view["Frecuencia relativa (hi)"] = df_fmt_view["Frecuencia relativa (hi)"].map(lambda x: f"{x:.4f}")
        df_fmt_view["Frecuencia porcentual (%)"] = df_fmt_view["Frecuencia porcentual (%)"].map(lambda x: f"{x:.2f}%")
        
        st.dataframe(df_fmt_view, use_container_width=True, hide_index=True)

        st.write("**Exportar Tabla (Incluye Gráficos):**")
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        
        df_export = preparar_tabla_exportacion(df)
        nombre_export = st.session_state.get('nombre_actual', 'Tabla')
        
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        col_e1.download_button("📄 CSV (Solo datos)", data=csv_data, file_name="tabla.csv", mime="text/csv", use_container_width=True)
        
        excel_data = generar_excel_tabla(df_export, info, nombre_export, datos_raw)
        col_e2.download_button("📊 Excel (Completo)", data=excel_data, file_name="reporte.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
        
        pdf_data = generar_pdf_tabla(df_export, info, nombre_export, datos_raw)
        col_e3.download_button("📕 PDF (Completo)", data=pdf_data, file_name="reporte.pdf", mime="application/pdf", use_container_width=True)
        
        png_data = generar_png_tabla(df_export, info, nombre_export, datos_raw)
        col_e4.download_button("🖼️ PNG (Completo)", data=png_data, file_name="reporte.png", mime="image/png", use_container_width=True)

        if st.button("💾 Guardar Tabla en Historial"):
            st.session_state.tablas_guardadas.append({"nombre": st.session_state.nombre_actual, "datos": datos_raw})
            guardar_tablas_en_disco()
            st.success(f"Tabla guardada permanentemente.")
    else:
        st.info("Aún no has procesado datos.")

# --- PESTAÑA 3: GRÁFICOS ---
with tab_charts:
    if st.session_state.df is not None and st.session_state.info is not None:
        df = st.session_state.df
        datos_raw = st.session_state.datos_procesados
        info = st.session_state.info
        colores = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C7C7C7', '#5366FF', '#289F40', '#D26384']

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_bar = px.bar(df, x="Intervalo de clase", y="Frecuencia absoluta (fi)", title="Gráfico de Barras", color="Intervalo de clase", color_discrete_sequence=colores, text="Frecuencia absoluta (fi)")
            fig_bar.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)

            if datos_raw:
                fig_hist = px.histogram(x=datos_raw, nbins=info['k'], title="Histograma Verdadero", color_discrete_sequence=['#72B7B2'])
                fig_hist.update_layout(template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)

        with col_g2:
            fig_pie = px.pie(df, values="Frecuencia absoluta (fi)", names="Intervalo de clase", title="Gráfico Circular", color_discrete_sequence=colores, hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Aún no has procesado datos.")

# --- PESTAÑA 4: TABLAS GUARDADAS ---
with tab_saved:
    if not st.session_state.tablas_guardadas:
        st.info("No hay tablas guardadas en el historial.")
    else:
        for i, tabla in enumerate(st.session_state.tablas_guardadas):
            with st.expander(f"📁 {tabla['nombre']}"):
                df_temp, info_temp = construir_tabla(tabla['datos'])
                df_fmt_temp = df_temp.copy()
                df_fmt_temp["Marca de clase (Xi)"] = df_fmt_temp["Marca de clase (Xi)"].map(lambda x: f"{x:.2f}")
                df_fmt_temp["Frecuencia relativa (hi)"] = df_fmt_temp["Frecuencia relativa (hi)"].map(lambda x: f"{x:.4f}")
                df_fmt_temp["Frecuencia porcentual (%)"] = df_fmt_temp["Frecuencia porcentual (%)"].map(lambda x: f"{x:.2f}%")
                
                st.dataframe(df_fmt_temp, use_container_width=True, hide_index=True)
                
                st.write("**Descargar esta tabla:**")
                c1, c2, c3, c4, c5 = st.columns(5)
                
                df_export_temp = preparar_tabla_exportacion(df_temp)
                nombre_export_temp = tabla['nombre']
                datos_raw_temp = tabla['datos']
                
                c1.download_button("📄 CSV", data=df_export_temp.to_csv(index=False).encode('utf-8'), file_name=f"{nombre_export_temp}.csv", mime="text/csv", key=f"csv_{i}")
                
                excel_data_temp = generar_excel_tabla(df_export_temp, info_temp, nombre_export_temp, datos_raw_temp)
                c2.download_button("📊 Excel", data=excel_data_temp, file_name=f"{nombre_export_temp}.xlsx", mime="application/vnd.ms-excel", key=f"xls_{i}")
                
                c3.download_button("📕 PDF", data=generar_pdf_tabla(df_export_temp, info_temp, nombre_export_temp, datos_raw_temp), file_name=f"{nombre_export_temp}.pdf", mime="application/pdf", key=f"pdf_{i}")
                c4.download_button("🖼️ PNG", data=generar_png_tabla(df_export_temp, info_temp, nombre_export_temp, datos_raw_temp), file_name=f"{nombre_export_temp}.png", mime="image/png", key=f"png_{i}")
                
                if c5.button("🗑️ Eliminar", key=f"del_{i}", use_container_width=True):
                    st.session_state.tablas_guardadas.pop(i)
                    guardar_tablas_en_disco()
                    st.rerun()