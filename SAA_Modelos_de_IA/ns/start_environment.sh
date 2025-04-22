#!/bin/bash
# COLOCAR UNA COPIA DE ESTE ARCHIVO EN LA CARPETA DEL PROYECTO A EJECUTAR 
# este script debe estar en la carpeta que se utilizará como carpeta de trabajo por la notebook que se levantará dentro del environment
# Rutas a donde están todos los environments y a donde está el enviroment que quiero activar con el proyecto que voy a trabajar
env_path="/home/alvaro/"
project_env="python_env1"
current_directory=$(pwd)
# activar el environment desde la carpeta de trabajo y levanta la notebook con este directorio de trabajo
cd $env_path/$project_env
./run_environment_jup_lab.sh $current_directory
