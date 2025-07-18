#!/bin/bash

set -e # Abortar o script se algún comando falla

# --- Variables de Configuración ---
APP_NAME="VideoConverterPro"
LOWERCASE_APP_NAME="videoconverterpro"
PROJECT_DIR=$(pwd)
APP_DIR="${PROJECT_DIR}/${APP_NAME}.AppDir"
ICON_PATH="${PROJECT_DIR}/icon.png"
# O nome do icono dentro da AppDir (sen extensión)
ICON_NAME_NO_EXT="videoconverterpro"
DESKTOP_FILE="${PROJECT_DIR}/${LOWERCASE_APP_NAME}.desktop"

echo "--- Iniciando a creación da AppImage (método robusto) ---"

# --- Limpeza ---
echo "Limpando compilacións anteriores..."
rm -rf "$APP_DIR" "${APP_NAME}-x86_64.AppImage" *.desktop

# --- Descarga de Ferramentas ---
echo "Descargando linuxdeploy..."
wget -c "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage" -O linuxdeploy
chmod +x ./linuxdeploy

# --- Estrutura da AppDir ---
echo "Creando a estrutura de directorios da AppDir..."
mkdir -p "$APP_DIR/usr/bin"
mkdir -p "$APP_DIR/usr/lib"
mkdir -p "$APP_DIR/usr/src"

# --- Instalación de Python e Dependencias ---
echo "Instalando Python e dependencias dentro da AppDir..."
python3 -m venv "$APP_DIR/usr/python_venv"
PYTHON_EXE_IN_VENV="$APP_DIR/usr/python_venv/bin/python"
SYSTEM_PYTHON=$(readlink -f "$PYTHON_EXE_IN_VENV")
echo "Copiando o binario de Python ($SYSTEM_PYTHON) ao venv..."
rm "$PYTHON_EXE_IN_VENV"
cp "$SYSTEM_PYTHON" "$PYTHON_EXE_IN_VENV"
source "$APP_DIR/usr/python_venv/bin/activate"
pip install -U pip wheel
pip install -r requirements.txt
deactivate

# --- Copia de Ficheiros da Aplicación ---
echo "Copiando o código fonte..."
cp -r src/* "$APP_DIR/usr/src/"

echo "Copiando o icono..."
# Copiar e renomear o icono para que coincida co .desktop
cp "$ICON_PATH" "$APP_DIR/${ICON_NAME_NO_EXT}.png"

echo "Creando o ficheiro .desktop..."
cat <<EOF > "$DESKTOP_FILE"
[Desktop Entry]
Name=$APP_NAME
Comment=Unha ferramenta para converter e procesar vídeos
Exec=AppRun
Icon=${ICON_NAME_NO_EXT}
Type=Application
Categories=AudioVideo;Video;
EOF
cp "$DESKTOP_FILE" "$APP_DIR/"

# --- Script de Lanzamento AppRun ---
echo "Creando o script AppRun..."
cat <<'EOF' > "$APP_DIR/AppRun"
#!/bin/bash
# Obter o directorio onde se atopa o script AppRun
HERE=$(dirname $(readlink -f "${0}"))
export APPDIR="$HERE"

# Gardar o directorio de traballo actual (dende onde se lanzou a AppImage)
export LAUNCH_CWD=$(pwd)

# Activar o contorno virtual de Python
source "$HERE/usr/python_venv/bin/activate"

# Engadir o directorio 'usr' da AppDir ao PYTHONPATH para que Python atope os módulos
export PYTHONPATH="$HERE/usr:$PYTHONPATH"

# Engadir as librarías da AppDir ao path
export LD_LIBRARY_PATH="$HERE/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Lanzando VideoConverterPro..."

# Cambiar ao directorio de traballo orixinal para que os ficheiros de saída se garden alí
cd "$LAUNCH_CWD"

# Executar o script principal de Python
# Usamos a ruta completa ao Python do venv para máxima portabilidade
"$HERE/usr/python_venv/bin/python" -m src.main --appdir "$APPDIR"

echo ""
echo "----------------------------------------------------"
echo "A execución rematou. Preme Intro para pechar."
echo "----------------------------------------------------"
read
EOF
chmod +x "$APP_DIR/AppRun"

# --- Empaquetado con linuxdeploy ---
echo "Buscando e empaquetando dependencias do sistema con linuxdeploy..."
./linuxdeploy --appdir "$APP_DIR" --output appimage \
    --deploy-deps-only "$APP_DIR/usr/python_venv/bin/python" \
    --icon-file "$APP_DIR/${ICON_NAME_NO_EXT}.png" \
    --desktop-file "$DESKTOP_FILE"

# --- Limpeza Final ---
echo "Limpando ficheiros temporais..."
rm -f linuxdeploy "$DESKTOP_FILE"

echo ""
echo "--- Proceso completado ---"
echo "AppImage creada con éxito: ${APP_NAME}-x86_64.AppImage"
echo "Para probala, abre un terminal e executa: ./${APP_NAME}-x86_64.AppImage"