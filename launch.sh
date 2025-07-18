#!/bin/bash

APPIMAGE_NAME="VideoConverterPro-x86_64.AppImage"
APPIMAGE_PATH="$(dirname "$(readlink -f "$0")")/$APPIMAGE_NAME"

# Comprobar se estamos a executar nunha terminal interactiva
if ! [ -t 1 ]; then
    # Non estamos nunha terminal (ex: dobre clic dende a GUI)
    
    # Buscar un emulador de terminal dispoñible
    TERMINAL_CMD=""
    if command -v gnome-terminal &>/dev/null; then
        TERMINAL_CMD="gnome-terminal --"
    elif command -v konsole &>/dev/null; then
        TERMINAL_CMD="konsole -e"
    elif command -v xfce4-terminal &>/dev/null; then
        TERMINAL_CMD="xfce4-terminal --hold -e"
    elif command -v mate-terminal &>/dev/null; then
        TERMINAL_CMD="mate-terminal -e"
    elif command -v xterm &>/dev/null; then
        # A opción -hold mantén o terminal aberto ao rematar
        TERMINAL_CMD="xterm -hold -e"
    fi

    if [ -n "$TERMINAL_CMD" ]; then
        # Atopouse un terminal. Volver a lanzar este mesmo script dentro del e saír.
        exec $TERMINAL_CMD "$APPIMAGE_PATH" "$@"
    else
        # Non se atopou ningún terminal. Informar ao usuario e saír.
        ERROR_MSG="Non se atopou un terminal compatible (gnome-terminal, konsole, xterm, etc.) para mostrar a saída.\n\nA aplicación non pode continuar.\n\nPor favor, instale un terminal (ex: 'sudo apt install xterm') ou execute a AppImage dende unha consola existente."
        if command -v zenity &>/dev/null; then
            zenity --error --text="$ERROR_MSG" --title="VideoConverterPro - Erro Crítico"
        elif command -v kdialog &>/dev/null; then
            kdialog --error "$ERROR_MSG" --title="VideoConverterPro - Erro Crítico"
        fi
        exit 1
    fi
else
    # Xa estamos nun terminal, executar a AppImage directamente
    exec "$APPIMAGE_PATH" "$@"
fi
