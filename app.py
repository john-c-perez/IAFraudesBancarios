# Importar el recolector de basura para liberar memoria cuando sea necesario
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

# Cargar el dataset normalizado y otros recursos
historical_data = pd.read_csv('clientes_filtrados_normales_normalizados.csv', sep=';')
cliente_stats_df = pd.read_csv('cliente_stats_escalarporcliente.csv', sep=',')
client_stats_normales = pd.read_csv('client_stats_normales.csv')  # Cargar estadísticas normales
with open('scaler_global.pkl', 'rb') as file:
    scaler_global = pickle.load(file)
discriminator = load_model('discriminator.h5')

# Crear el mapeo de los valores únicos de Cod_Cliente
cliente_map = {v: i for i, v in enumerate(historical_data['Cod_Cliente'].unique())}
historical_data['Cod_Cliente'] = historical_data['Cod_Cliente'].map(cliente_map)
cliente_stats_df.set_index('Cod_Cliente', inplace=True)
column_order = [
    'Cod_Cliente', 'Valor',
    'Transaccion_PAGO TARJETA DE CREDITO', 'Transaccion_TRANSFER BANCOS',
    'Transaccion_TRANSFER EXTERNAS', 'Transaccion_TRANSFER INTERNAS',
    'Segmento_Hora_Madrugada', 'Segmento_Hora_Mañana', 'Segmento_Hora_Noche',
    'Segmento_Hora_Tarde', 'Segmento_Dia_Mes_Final de Mes',
    'Segmento_Dia_Mes_Mitad de Mes', 'Segmento_Dia_Mes_Principio de Mes'
    ]

@app.route('/')
def index():
    # Obtener la lista de clientes únicos
    top_clients = client_stats_normales['Cod_Cliente'].unique().tolist()
    return render_template('index.html', top_clients=top_clients)

@app.route('/predict', methods=['POST'])
def evaluar_transaccion():
    # Recibir datos en formato JSON
    data = request.json

    if not data:
        return jsonify({"error": "No se recibieron datos en la solicitud"}), 400

    # Extraer valores
    cod_cliente = int(data['Cod_Cliente'])
    hora = data['Horas']
    fecha = data['Fecha']
    transaccion = data['Transaccion']
    valor = float(data['Valor'])

    # Preprocesar la transacción
    data = pd.DataFrame({
        'Cod_Cliente': [cod_cliente],
        'Horas': [hora],
        'Fecha': [fecha],
        'Transaccion': [transaccion],
        'Valor': [valor]
    })

    # Segmentación y preprocesamiento
    data['Segmento_Hora'] = data['Horas'].apply(segmentar_horas)
    data['Segmento_Dia_Mes'] = data['Fecha'].apply(segmentar_dia_mes)
    data = data.drop(columns=['Horas', 'Fecha'])
    data = pd.get_dummies(data, columns=['Transaccion', 'Segmento_Hora', 'Segmento_Dia_Mes'], drop_first=False)

    # Normalizar el valor de la transacción
    valor_min = float(cliente_stats_df.loc[cod_cliente, 'min'])
    valor_max = float(cliente_stats_df.loc[cod_cliente, 'max'])
    valor_normalizado = 2 * (data['Valor'].iloc[0] - valor_min) / (valor_max - valor_min) - 1
    
    data['Cod_Cliente'] = data['Cod_Cliente'].map(cliente_map)
    # Asegurar que todas las columnas estén presentes y ordenadas
    # Definir el orden deseado de las columnas
    
    columnas_originales = scaler_global.feature_names_in_
    # Asegurar que todas las columnas dummies están presentes y ordenar correctamente
    for col in columnas_originales:
        if col not in data.columns:
            data[col] = 0
    # Ordenar las columnas según el orden original
    data = data[columnas_originales]  
        # Normalizar las columnas usando el scaler global
    data = pd.DataFrame(scaler_global.transform(data), columns=columnas_originales)
    data['Valor'] = valor_normalizado
    data['Cod_Cliente'] = cliente_map[cod_cliente]
    # Asegurarse de que todas las columnas necesarias están presentes
    for col in column_order:
        if col not in data.columns:
            data[col] = 0

    # Reorganizar las columnas del DataFrame en el orden correcto
    data = data[column_order] 
    
    # Evaluar la transacción
    prediction = discriminator.predict([data.values[:, 1:], data['Cod_Cliente'].values])

    # Evaluación con umbrales dinámicos
    scores_historicos = obtener_puntuaciones_historicas(cliente_map[cod_cliente], historical_data, discriminator)
    reglas_dinamicas = definir_reglas_dinamicas_con_flexibilidad(scores_historicos, n_clusters=3, tolerancia=0.01)
    clasificacion = evaluar_transaccion_con_flexibilidad(prediction, reglas_dinamicas)

    # Extraer estadísticas del cliente para la explicación
    cliente_stats = client_stats_normales[client_stats_normales['Cod_Cliente'] == cod_cliente]
    
    # Convertir tipos de datos a nativos de Python
    detalles = {
        'Cod_Cliente': int(cliente_stats['Cod_Cliente'].values[0]),  # Convertir a tipo Python
        'Max_Valor': float(cliente_stats['Max_Valor'].values[0]),     # Convertir a tipo Python
        'Num_Transacciones': int(cliente_stats['Num_Transacciones'].values[0]),  # Convertir a tipo Python
        'Hora_Mas_Frecuente': cliente_stats['Hora_Mas_Frecuente'].values[0],
        'Dia_Mas_Frecuente': cliente_stats['Dia_Mas_Frecuente'].values[0],
        'Transaccion_Mas_Frecuente': cliente_stats['Transaccion_Mas_Frecuente'].values[0]
    }

    # Generar la explicación si la transacción es anómala
    explicacion = ""
    hora_segmento = segmentar_horas(hora)
    dia_segmento = segmentar_dia_mes(fecha)

    # Eliminar el prefijo "Ocurrencias_" en las explicaciones
    detalles['Hora_Mas_Frecuente'] = detalles['Hora_Mas_Frecuente'].replace('Ocurrencias_', '')
    detalles['Dia_Mas_Frecuente'] = detalles['Dia_Mas_Frecuente'].replace('Ocurrencias_', '')
    detalles['Transaccion_Mas_Frecuente'] = detalles['Transaccion_Mas_Frecuente'].replace('Ocurrencias_', '')

    # Comparar los segmentos de hora
    if hora_segmento != detalles['Hora_Mas_Frecuente']:
        explicacion += f"Cliente suele hacer transacciones en {detalles['Hora_Mas_Frecuente']}, pero esta fue en {hora_segmento}. "

    # Comparar los segmentos de día del mes
    if dia_segmento != detalles['Dia_Mas_Frecuente']:
        explicacion += f"Cliente suele hacer transacciones en {detalles['Dia_Mas_Frecuente']}, pero esta fue en {dia_segmento}. "

    # Comparar el tipo de transacción
    if transaccion != detalles['Transaccion_Mas_Frecuente']:
        explicacion += f"Cliente suele realizar {detalles['Transaccion_Mas_Frecuente']}, pero esta fue una {transaccion}. "

    # Comparar el valor de la transacción
    if valor > detalles['Max_Valor']:
        explicacion += f"El valor máximo de transacción para este cliente es {detalles['Max_Valor']}, pero esta fue de {valor}. "
    
    # Limpiar la explicación final
    explicacion = explicacion.strip()

    # Imprimir detalles para depuración
    print(f"Detalles de la transacción:")
    print(f"  - Cliente: {cod_cliente}")
    print(f"  - Valor: {valor}")
    print(f"  - Transacción: {transaccion}")
    print(f"  - Hora: {hora}")
    print(f"  - Fecha: {fecha}")
    print(f"  - Clasificación: {clasificacion}")
    print(f"  - Estadísticas del cliente: {detalles}")
    # Solo incluir la explicación si la transacción no es NORMAL
    response = {
        'clasificacion': clasificacion
    }

    if clasificacion != "NORMAL":
        response['explicacion'] = explicacion

    # Devolver la respuesta JSON
    return jsonify(response)

def segmentar_horas(hora_str):
    hora = int(hora_str.split(':')[0])
    if 0 <= hora < 6:
        return 'Madrugada'
    elif 6 <= hora < 12:
        return 'Mañana'
    elif 12 <= hora < 18:
        return 'Tarde'
    else:
        return 'Noche'

def segmentar_dia_mes(fecha_str):
    # Verificar si el formato es "YYYY-MM-DD"
    if '-' in fecha_str:
        dia = int(fecha_str.split('-')[2])  # Extrae el día del formato "YYYY-MM-DD"
    else:
        dia = int(fecha_str.split('/')[0])  # Extrae el día del formato "DD/MM/YYYY"
    
    if 1 <= dia <= 10:
        return 'Principio de Mes'
    elif 11 <= dia <= 20:
        return 'Mitad de Mes'
    else:
        return 'Final de Mes'

def obtener_puntuaciones_historicas(cod_cliente, historical_data, discriminator):
    cliente_data = historical_data[historical_data['Cod_Cliente'] == cod_cliente]
    scores = discriminator.predict([cliente_data.values[:, 1:], cliente_data['Cod_Cliente'].values])
    return scores

def definir_reglas_dinamicas_con_flexibilidad(scores_historicos, n_clusters=3, tolerancia=0.01):
    if np.all(scores_historicos == scores_historicos[0]):
        reglas = {
            "anómalo": scores_historicos[0],
            "sospechoso": scores_historicos[0]+ tolerancia * 5,
            "flexible": scores_historicos[0]+ tolerancia * 2,
            "normal": scores_historicos[0]+ tolerancia * 5,
        }
        return reglas

    scores_historicos = np.array(scores_historicos).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(scores_historicos)
    clusters = kmeans.predict(scores_historicos)
    cluster_limits = {i: (scores_historicos[clusters == i].min(), scores_historicos[clusters == i].max()) for i in range(n_clusters)}
    sorted_clusters = sorted(cluster_limits.items(), key=lambda x: x[1][1])
    
    reglas = {
        "anómalo": sorted_clusters[-1][1][1],
        "sospechoso": sorted_clusters[-1][1][1] - tolerancia * 25,
        "flexible": sorted_clusters[-2][1][1],
        "normal": sorted_clusters[-3][1][1]
    }
    return reglas

def evaluar_transaccion_con_flexibilidad(prediction, reglas, tolerancia=1e-3):
    if prediction < reglas["normal"]:
        return "NORMAL"
    elif prediction < reglas["flexible"]:
        return "NORMAL (FLEXIBLE)"
    elif prediction < reglas["sospechoso"]:
        return "SOSPECHOSO"
    else:
        return "ANÓMALO"
    


    ########################################
    #          EVALUACION DE CSV           #
    ########################################

@app.route('/cargar_csv', methods=['POST'])
def cargar_csv():
    # Recibir el archivo CSV cargado
    file = request.files['file']
    if not file:
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    # Cargar el dataset desde el archivo CSV
    dataset = pd.read_csv(file, sep=';')

    # Normalizar el dataset por lotes
    batch_size = 1000  # Puedes ajustar el tamaño del lote según la capacidad de tu sistema
    total_filas = len(dataset)
    etiquetas_predichas = []
    
    for start in range(0, total_filas, batch_size):
        end = min(start + batch_size, total_filas)
        batch = dataset.iloc[start:end]

        # Normalizar el batch
        columnas_a_normalizar = batch.columns.drop(['Cod_Cliente', 'Valor'])
        batch[columnas_a_normalizar] = scaler_global.transform(batch[columnas_a_normalizar])

        for cliente in batch['Cod_Cliente'].unique():
            valor_min = float(cliente_stats_df.loc[cliente, 'min'])
            valor_max = float(cliente_stats_df.loc[cliente, 'max'])
            batch.loc[batch['Cod_Cliente'] == cliente, 'Valor'] = (
                2 * (batch.loc[batch['Cod_Cliente'] == cliente, 'Valor'] - valor_min) / (valor_max - valor_min) - 1
            )

        batch['Cod_Cliente'] = batch['Cod_Cliente'].map(cliente_map)

        # Evaluar el batch
        for cliente in batch['Cod_Cliente'].unique():
            cliente_data = batch[batch['Cod_Cliente'] == cliente]
            predictions = discriminator.predict([cliente_data.drop(columns=['Cod_Cliente']).values, np.array([cliente] * len(cliente_data))])

            predictions = np.where(predictions == 0.0, predictions + 0.0001, predictions)
            umbral_dinamico = np.mean(predictions)

            for prediction in predictions:
                if prediction > umbral_dinamico:
                    etiquetas_predichas.append((cliente, 'ANÓMALO'))
                else:
                    etiquetas_predichas.append((cliente, 'NORMAL'))

    # Análisis de los resultados
    clientes_anomalos = pd.DataFrame(etiquetas_predichas, columns=['Cod_Cliente', 'Clasificación'])
    clientes_anomalos = clientes_anomalos[clientes_anomalos['Clasificación'] == 'ANÓMALO']
    top_clientes_anomalos = clientes_anomalos['Cod_Cliente'].value_counts().head(10)

    # Calcular la distribución de las transacciones
    distribucion_transacciones = {
        'Normal': len([x for _, x in etiquetas_predichas if x == 'NORMAL']),
        'Anómalo': len([x for _, x in etiquetas_predichas if x == 'ANÓMALO'])
    }

    # Resumen de los resultados
    total_datos = len(etiquetas_predichas)
    total_normales = distribucion_transacciones['Normal']
    total_anomalos = distribucion_transacciones['Anómalo']
    precision_atipicos = total_anomalos / total_datos if total_datos > 0 else 0

    # Enviar resultados al frontend
    return jsonify({
        "total_datos": total_datos,
        "total_normales": total_normales,
        "total_anomalos": total_anomalos,
        "precision_atipicos": precision_atipicos * 100,
        "clientes_anomalos": top_clientes_anomalos.to_dict(),
        "distribucion_transacciones": distribucion_transacciones
    })



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5000))
