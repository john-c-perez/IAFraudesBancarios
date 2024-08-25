# IAFraudesBancarios
Hola, esta es un prototipo  de modelo de deep learning con CGAN para deteccion de fraudes bancarios hecho con tensorflow.

para importar libreiras abre requirements.txt incluye librerias y la linea que debes ejecutar.
si no quieres instalarlas globalmente en tu dispositivo puedes crear un Entorno virtual
crea dentro de donde esta tu proyecto un ambiente virtual y aqui ejecuta

1. Comprobar con el Comando pip list sin un Entorno Virtual Activo
Si no tienes un entorno virtual activo y ejecutas el comando pip list, verás todos los paquetes que están instalados globalmente en tu sistema Python.

Para hacer esto:

Abre el terminal en VSCode:

Ve a Terminal > New Terminal en el menú superior.
Ejecuta el comando pip list:
pip list --format=columns o
pip list unicamente.

Verifica la ruta de python: Para asegurarte de que no estás en un entorno virtual, puedes ejecutar:
which python
(En macOS/Linux) o:
where python
(En Windows).

2. Verificar la Ubicación de los Paquetes Instalados
Puedes verificar dónde está instalado un paquete en particular usando el siguiente comando:
pip show <nombre_del_paquete>

Este comando muestra información sobre el paquete, incluyendo su ubicación. Si la ubicación es en un directorio como site-packages dentro de la instalación principal de Python (por ejemplo, /usr/local/lib/python3.x/site-packages en macOS/Linux o C:\Python\Python39\Lib\site-packages en Windows), entonces el paquete está instalado globalmente.


Crear un entorno virtual en Python es una práctica común y recomendada para gestionar dependencias de proyectos. Aquí te explico cómo hacerlo paso a paso en diferentes sistemas operativos usando Visual Studio Code (VSCode).

1. Asegúrate de Tener venv Instalado
venv es una herramienta incluida en las versiones modernas de Python (3.3 en adelante). No necesitas instalar nada adicional si ya tienes Python instalado.
2. Abrir Visual Studio Code (VSCode)
Abre VSCode y navega al directorio de tu proyecto o crea una nueva carpeta para tu proyecto.
3. Abrir el Terminal Integrado en VSCode
Ve a Terminal > New Terminal en el menú superior o usa el atajo Ctrl + (en Windows/Linux) o Cmd + (en macOS).
4. Crear el Entorno Virtual
Una vez en el terminal, asegúrate de estar en la carpeta de tu proyecto y ejecuta uno de los siguientes comandos:

en windows:
python -m venv nombre_del_entorno
En macOS/Linux:
python3 -m venv nombre_del_entorno

5. Activar el Entorno Virtual
Después de crear el entorno virtual, necesitarás activarlo.

en windows:
.\nombre_del_entorno\Scripts\activate
En macOS/Linux:
source nombre_del_entorno/bin/activate

Después de activar el entorno virtual, deberías ver el nombre del entorno entre paréntesis al principio de tu terminal, lo que indica que el entorno virtual está activo. Por ejemplo: (venv) C:\path\to\your\project>.
para activar el entorno virtual:
en la ubicacion de la carpeta de tu proyecto coloca
C:\path\to\your\project\(venv)\Scripts\activate

6. Instalar Paquetes en el Entorno Virtual
Con el entorno virtual activado, cualquier paquete que instales con pip se instalará localmente dentro de ese entorno, no afectando las instalaciones globales.
pip install nombre_del_paquete
pip uninstall nombre_del_paquete
7. Desactivar el Entorno Virtual
Cuando termines de trabajar, puedes desactivar el entorno virtual ejecutando:
C:\path\to\your\project\(venv)\Scripts\desactivate
