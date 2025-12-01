#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:53:11 2024

@author: cristiantobar
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import scipy.cluster.hierarchy as shc


def aglom_clustering(df):
    df.head()
    #df_cont = df.drop(['id'], axis = 1)
    df_cont_encoded = pd.get_dummies(df, columns=['typology'])
    df.head()
    data_scaled = normalize(df_cont_encoded)
    data_scaled = pd.DataFrame(data_scaled, columns=df_cont_encoded.columns)
    data_scaled.head()
    
    # plt.figure(figsize=(7, 3))  
    # plt.title("Dendrograma")  
    # dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    # plt.savefig('db1_dendogram.pdf',bbox_inches='tight')
    # plt.show()
    # # Configuración de estilo
    # sns.set_theme(style="white", font_scale=1.2)
    
    # Crear la figura
    # plt.figure(figsize=(7, 3))
    # plt.title("Jerarquía de similitud entre proyectos", fontsize=13, weight='bold', pad=15)
    
    # # Dendrograma con estilo mejorado
    # dend = shc.dendrogram(
    #     shc.linkage(data_scaled, method='ward'),
    #     color_threshold=0.7 * max(shc.linkage(data_scaled, method='ward')[:, 2]),  # corte visual
    #     above_threshold_color='gray',     # color de ramas por encima del umbral
    #     #below_threshold_color='#4C72B0',  # color de ramas por debajo
    #     leaf_rotation=90,                 # etiquetas más legibles
    #     leaf_font_size=10,                # tamaño de fuente
    #     distance_sort='descending',       # ramas ordenadas por distancia
    #     show_leaf_counts=True             # muestra conteo de elementos en cada grupo
    # )
    
    # # Decoración estética
    # plt.xlabel("Observaciones (proyectos)", fontsize=11)
    # plt.ylabel("Distancia euclidiana", fontsize=11)
    # plt.grid(axis='y', linestyle='--', alpha=0.4)
    # sns.despine(left=False, bottom=False)
    
    # # Guardar y mostrar
    # plt.tight_layout()
    # plt.savefig('db1_dendrograma_mejorado.pdf', dpi=600, bbox_inches='tight')
    # plt.show()
    
    cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')  
    y_predict = cluster.fit_predict(data_scaled)
    
    #Validar los datos para probar si está asignando bien
    # Escalar solo las numericas y las categoricas, no. Porque eso puede afectar.
    
    # plt.scatter(df_cont_encoded['weeks_duration'], df_cont_encoded['weeks_delay'], c=y_predict)
    # plt.xlabel('\n weeks \n')
    # plt.ylabel('\n delay weeks \n')
    # plt.show()
    
    # plt.scatter(df_cont_encoded['built_area'], df_cont_encoded['weeks_delay'], c=y_predict)
    # plt.xlabel('\n Squared meters \n')
    # plt.ylabel('\n delay weeks  \n')
    # plt.show()
    
    # plt.scatter(df_cont_encoded['modul_price'], df_cont_encoded['weeks_delay'], c=y_predict)
    # plt.xlabel('\n $ per squared meter \n')
    # plt.ylabel('\n delay weeks  \n')
    # plt.show()
    
    # plt.scatter(df_cont_encoded['weeks_delay'], df_cont_encoded['weeks_delay'], c=y_predict)
    # plt.xlabel('\n delay weeks \n')
    # plt.ylabel('\n delay weeks \n')
    # plt.show() 
   
    # plt.scatter(data_scaled['weeks_duration'], data_scaled['weeks_delay'], c=y_predict)
    # plt.xlabel('\n weeks \n')
    # plt.ylabel('\n delay weeks \n')
    # plt.show()
    
    # plt.scatter(data_scaled['built_area'], data_scaled['weeks_delay'], c=y_predict)
    # plt.xlabel('\n Squared meters \n')
    # plt.ylabel('\n delay weeks  \n')
    # plt.show()
    
    # plt.scatter(data_scaled['modul_price'], data_scaled['weeks_delay'], c=y_predict)
    # plt.xlabel('\n $ per squared meter \n')
    # plt.ylabel('\n delay weeks  \n')
    # plt.show()
    
    # plt.scatter(data_scaled['weeks_delay'], data_scaled['weeks_delay'], c=y_predict)
    # plt.xlabel('\n delay weeks \n')
    # plt.ylabel('\n delay weeks \n')
    # plt.show() 
    
        
    y_predict = pd.DataFrame(y_predict)
    y_predict.columns = ['output'] #1 para descarado, 0 para no descarado
    

    
    df_temp = df_cont_encoded.reset_index()
    df_cont_encoded_res = df_temp.drop(['index'], axis = 1).copy()
    
    new_df = pd.concat([df_cont_encoded_res, y_predict], axis = 1)
        
    # plt.subplots(figsize=(7,4)) 
    # sns.boxplot(x=new_df['weeks_duration'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('Duración del proyecto')
    # plt.xlabel('Semanas')
    # plt.savefig('db1_1_weeks_duration.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,4)) 
    # sns.boxplot(x=new_df['built_area'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('Área construida')
    # plt.xlabel('metros cuadrados')
    # plt.savefig('db1_2_built_area.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,4)) 
    # sns.boxplot(x=new_df['modul_price'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('Precio por metro cuadrado')
    # plt.xlabel('Dólares')
    # plt.savefig('db1_3_modul_price.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,4)) 
    # sns.boxplot(x=new_df['weeks_delay'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('Semanas de retraso')
    # plt.xlabel('Semanas')
    # plt.savefig('db1_4_weeks_delay.pdf',bbox_inches='tight')
    # plt.show()
    
    # # Configuración general
    # sns.set(style="whitegrid", font_scale=1.2)
    
    # # Crear la figura y los ejes (1 fila, 4 columnas)
    # fig, axes = plt.subplots(1, 4, figsize=(10, 3.5))  # Ajusta ancho según lo que admita tu journal
    
    # # Paleta común
    # palette = 'Set2'
    
    # # --- 1. Duración del proyecto ---
    # sns.boxplot(x=new_df['weeks_duration'], hue=pd.Categorical(new_df.output), width=0.5,
    #             palette=palette, ax=axes[0])
    # axes[0].set_title('Duración del proyecto', fontsize=11)
    # axes[0].set_xlabel('Semanas', fontsize=10)
    # axes[0].set_ylabel('')
    # axes[0].legend_.remove()  # quitar leyenda repetida
    
    # # --- 2. Área construida ---
    # sns.boxplot(x=new_df['built_area'], hue=pd.Categorical(new_df.output), width=0.5,
    #             palette=palette, ax=axes[1])
    # axes[1].set_title('Área construida', fontsize=11)
    # axes[1].set_xlabel('Metros cuadrados', fontsize=10)
    # axes[1].set_ylabel('')
    # axes[1].legend_.remove()
    
    # # --- 3. Precio por metro cuadrado ---
    # sns.boxplot(x=new_df['modul_price'], hue=pd.Categorical(new_df.output), width=0.5,
    #             palette=palette, ax=axes[2])
    # axes[2].set_title('Precio por metro cuadrado', fontsize=11)
    # axes[2].set_xlabel('Dólares', fontsize=10)
    # axes[2].set_ylabel('')
    # axes[2].legend_.remove()
    
    # # --- 4. Semanas de retraso ---
    # sns.boxplot(x=new_df['weeks_delay'], hue=pd.Categorical(new_df.output), width=0.5,
    #             palette=palette, ax=axes[3])
    # axes[3].set_title('Semanas de retraso', fontsize=11)
    # axes[3].set_xlabel('Semanas', fontsize=10)
    # axes[3].set_ylabel('')
    # axes[3].legend_.remove()

    
    # # Ajustes finales
    # for ax in axes:
    #     sns.despine(ax=ax)
    #     ax.tick_params(axis='x', labelrotation=0)
    #     ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # # Leyenda única centrada
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, title='Clase', loc='upper center', ncol=2, fontsize=10, frameon=False)
    
    # plt.tight_layout(rect=[0, 0, 1, 0.92])
    # plt.savefig('boxplots_variables_articulo.pdf', dpi=600, bbox_inches='tight')
    # plt.show()
    
    # ----------------- CONFIGURACIÓN GENERAL -----------------
    sns.set(style="whitegrid", font_scale=1.2)
    palette = 'Set2'
    palette2 = {0: '#fc8d62', 1: '#66c2a5'}
    # Crear figura con 1 fila y 5 columnas
    fig, axes = plt.subplots(1, 5, figsize=(12, 4))  # ancho proporcional al contenido
    
    # ----------------- (a) DENDROGRAMA -----------------
    linkage_matrix = shc.linkage(data_scaled, method='ward')
    dend = shc.dendrogram(
        linkage_matrix,
        color_threshold=0.7 * max(linkage_matrix[:, 2]),
        #above_threshold_color='gray',
        #above_threshold_color='#E69F00',   # naranja (antes gris)
        #below_threshold_color='#009E73',   # verde (antes azul)
        leaf_rotation=90,
        leaf_font_size=9,
        distance_sort='descending',
        show_leaf_counts=True,
        ax=axes[0]
    )
    
    axes[0].set_title('Jerarquía de similitud', fontsize=11, weight='bold')
    axes[0].set_xlabel('Proyectos', fontsize=10)
    axes[0].set_ylabel('Distancia euclidiana', fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    sns.despine(ax=axes[0], left=False, bottom=False)
    
    # ----------------- (b–e) BOXPLOTS -----------------
    
    # Lista de variables a graficar y etiquetas
    vars_to_plot = [
        ('weeks_duration', 'Duración del proyecto', 'Semanas'),
        ('built_area', 'Área construida', 'm²'),
        ('modul_price', 'Precio por m²', 'USD'),
        ('weeks_delay', 'Semanas de retraso', 'Semanas')
    ]
    
    for i, (var, title, xlabel) in enumerate(vars_to_plot):
        ax = axes[i+1]
        sns.boxplot(x=new_df[var], hue=pd.Categorical(new_df.output), width=0.5,
                    palette=palette2, ax=ax)
        ax.set_title(title, fontsize=11, weight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('')
        ax.legend_.remove()
        sns.despine(ax=ax)
        ax.tick_params(axis='x', labelrotation=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # ----------------- ETIQUETAS (a–e) -----------------
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    for label, ax in zip(labels, axes):
        ax.text(0.02, -0.25, label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')
    
    # ----------------- LEYENDA GLOBAL -----------------
    handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(handles, lbls, title='Clase', loc='upper center', ncol=2,
               fontsize=10, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('fig_dendrograma_boxplots.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    
    return new_df


    
