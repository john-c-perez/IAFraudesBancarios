$(document).ready(function () {
    bsCustomFileInput.init();  // Inicializar los inputs de archivos personalizados

    let chart;

    $('#transactionForm').on('submit', function (e) {
        e.preventDefault();  // Prevenir el comportamiento predeterminado del formulario

        const date = new Date($('#Fecha').val());
        const dayOfWeek = date.getUTCDay();  // Obtener el día de la semana
        const month = date.getUTCMonth() + 1;  // Obtener el mes (sumar 1 porque es base 0)
        const dayOfMonth = date.getUTCDate();  // Obtener el día del mes

        // Enviar la solicitud al backend para predecir si la transacción es anómala
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                Horas: $('#Horas').val(),
                Fecha: $('#Fecha').val(),
                Cod_Cliente: $('#Cod_Cliente').val(),
                Transaccion: $('#Transaccion').val(),
                Valor: $('#Valor').val()
            }),
            success: function (response) {
                console.log("Success:", response);
                $('#result').html(`<div class="alert alert-info">
                            Clasificación: ${response.clasificacion}<br>
                            Explicación: ${response.explicacion}
                          </div>`).fadeIn();

    
            },
            error: function (response) {
                $('#result').html(`<div class="alert alert-danger">Error: ${response.responseJSON.message}</div>`).fadeIn();
            }
        });
    });

    $('#csvForm').on('submit', function (e) {
        e.preventDefault();  // Prevenir el comportamiento predeterminado del formulario
        $('#progressBar').show();  // Mostrar la barra de progreso
        var formData = new FormData(this);  // Crear un objeto FormData con los datos del formulario

        // Enviar la solicitud al backend para cargar el archivo CSV
        $.ajax({
            url: '/cargar_csv',
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                let porcentajeAnomalos = parseFloat(response['precision_atipicos']);
                if (!isNaN(porcentajeAnomalos)) {
                    porcentajeAnomalos = porcentajeAnomalos.toFixed(2);
                } else {
                    porcentajeAnomalos = "No disponible";
                }

                let resultados = `
                    <div class="alert alert-info"><span class="alert-title">Total de datos evaluados:</span> ${response['total_datos']}</div>
                    <div class="alert alert-info"><span class="alert-title">Datos detectados como anómalos:</span> ${response['total_anomalos']}</div>
                    <div class="alert alert-info"><span class="alert-title">Datos detectados como normales:</span> ${response['total_normales']}</div>
                    <div class="alert alert-info"><span class="alert-title">Porcentaje de anómalos:</span> ${porcentajeAnomalos}%</div>
                    <div class="alert alert-info"><span class="alert-title">Clientes con más anomalías:</span></div>
                    <ul>
                `;

                // Iterar sobre los clientes con más anomalías y agregarlos al resultado
                Object.entries(response['clientes_anomalos']).forEach(function ([cliente, transacciones]) {
                    resultados += `<li><strong>Cliente:</strong> ${cliente}, <strong>Transacciones anómalas:</strong> ${transacciones}</li>`;
                });

                resultados += '</ul>';
                $('#csvResult').html(resultados).fadeIn();  // Mostrar los resultados del CSV

                updateChart(response);  // Actualizar el gráfico con los nuevos datos
                displayAdditionalInfo(response);  // Mostrar la información adicional
            },
            error: function (response) {
                $('#csvResult').html(`<div class="alert alert-danger">Error al cargar el archivo CSV.</div>`).fadeIn();
            },
            complete: function () {
                $('#progressBar').hide();  // Ocultar la barra de progreso
            }
        });
    });

    // Función para actualizar el gráfico de resultados
    function updateChart(response) {
        const ctx = document.getElementById('myChart').getContext('2d');

        if (chart) {
            chart.destroy();  // Destruir el gráfico previo si existe
        }

        // Crear un nuevo gráfico de barras
        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Transacciones Analizadas', 'Transacciones Anómalas', 'Porcentaje de Anomalías'],
                datasets: [{
                    label: 'Resultados',
                    data: [response['total_datos'], response['total_anomalos'], response['precision_atipicos']],
                    backgroundColor: [
                        '#C82333',
                        '#dc3545',
                        '#ffc107'
                    ],
                    borderColor: [
                        '#A71D2A',
                        '#bd2130',
                        '#d39e00'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true  // Comenzar el eje Y en cero
                    }
                }
            }
        });
    }

    // Función para mostrar información adicional
    function displayAdditionalInfo(response) {
        const clientesConMasAnomalias = Object.entries(response['clientes_anomalos']).map(
            ([cliente, transacciones]) => `<li><strong>Cliente ${cliente}:</strong> ${transacciones} transacciones anómalas</li>`
        ).join('');
    
        const distribucionTransacciones = Object.entries(response['distribucion_transacciones']).map(
            ([tipo, cantidad]) => `<li><strong>${tipo}:</strong> ${cantidad}</li>`
        ).join('');
    
        const additionalInfo = `
            <h4>Información Adicional del CSV</h4>
            <ul>
                <li><strong>Clientes con Más Transacciones Anómalas:</strong>
                    <ul>${clientesConMasAnomalias}</ul>
                </li>
                <li><strong>Distribución de Tipos de Transacciones:</strong>
                    <ul>${distribucionTransacciones}</ul>
                </li>
            </ul>
        `;
        $('#additionalInfo').html(additionalInfo).fadeIn();  // Mostrar la información adicional
    }
});
