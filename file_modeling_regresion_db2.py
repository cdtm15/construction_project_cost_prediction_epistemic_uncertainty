#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:14:50 2024

@author: cristiantobar
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import shap



#def feature_importance(df, nature):
def feature_importance(df, nature, top_n=None, save_tiff=False):

   #  X = df.drop('actual_construction_cost', axis=1).copy() # alternatively: X = df_no_missing.iloc[:,:-1]
   #  y = df['actual_construction_cost'].copy()

   #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   #  # Crear y entrenar el modelo
   #  model    = xgb.XGBRegressor(objective="reg:linear", random_state=42)
   #  model.fit(X_train, y_train)
    
   #  y_predict = model.predict(X_test)
   #  accu      = model.score(X_test, y_test)
   #  r2        = r2_score(y_test, y_predict)
    
   #  # 1) Importancias desde XGBoost (gain)
   #  gain = model.get_booster().get_score(importance_type='gain')
   #  imp_df = pd.DataFrame({'feature': list(gain.keys()), 'gain': list(gain.values())})
   #  imp_df = imp_df.sort_values('gain', ascending=False)
    
   #  # # (Opcional) normalizar a porcentajes para legibilidad
   # # imp_df['gain_pct'] = 100 * imp_df['gain'] / imp_df['gain'].sum()
   
   #  vals = imp_df['gain'].values
   #  labels_val = [f"{v:.2f}" for v in vals]    # cambia a f"{v:.2f}" si prefieres valor absoluto
    
   #  # 2) Plot
   #  fig, ax = plt.subplots(figsize=(12, max(4, 0.35*len(imp_df))))  # alto dinámico según #features
   #  y = range(len(imp_df))
    
   #  ax.barh(y, vals, align='center')
   #  ax.set_yticks(y)
   #  ax.set_yticklabels(imp_df['feature'], fontsize=11)
   #  ax.invert_yaxis()  # más importante arriba
    
   #  ax.set_xlabel("Feature importance (gain)", fontsize=12)
   #  ax.set_ylabel("Features", fontsize=12)
   #  ax.set_title("XGBoost Feature Importance (gain)", fontsize=13, weight='bold')
    
   #  # 3) Margen a la derecha y ubicación inteligente de etiquetas
   #  max_val = vals.max() if len(vals) else 1.0
   #  pad = 0.02 * max_val
   #  ax.set_xlim(0, max_val * 1.12)  # aire extra a la derecha
    
   #  for i, v in enumerate(vals):
   #      # Si la barra ocupa >90% del máximo, etiqueta por dentro en blanco; si no, por fuera
   #      inside = v > 0.90 * max_val
   #      if inside:
   #          ax.text(v - pad, i, labels_val[i],
   #                  va='center', ha='right', fontsize=10, color='white')
   #      else:
   #          ax.text(v + pad, i, labels_val[i],
   #                  va='center', ha='left', fontsize=10, color='black', clip_on=False)
    
   #  # 4) Grid sutil y ajuste de bordes
   #  ax.xaxis.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
   #  plt.tight_layout()
    
   #  # 5) Guardar en formatos de revista
   #  plt.savefig('feature_importance_external_'+nature+'.pdf',bbox_inches='tight')
   #  #plt.savefig("feature_importance_gain.tiff", dpi=300, bbox_inches="tight")
   #  plt.show()
    
   #  #Obtener la importancia de variables y visualizar
   #  fig, ax  = plt.subplots(figsize=(15, 8))
   #  plot_importance(model, importance_type='gain', grid= False, values_format='{v:.3f}', ax=ax)  # O 'weight' o 'cover' según lo necesites
   #  plt.savefig('feature_importance_db2_'+nature+'.pdf',bbox_inches='tight')
   #  plt.show()    
    
    """
   Entrena XGBoost, grafica importancia por gain y gráficos SHAP:
   - SHAP summary (dot)
   - SHAP summary (bar)
   - SHAP dependence plot para la feature más importante
   Args
   ----
   df : pd.DataFrame (incluye 'actual_construction_cost')
   nature : str        (sufijo de archivo)
   top_n : int|None    (si se indica, limita a top_n features en las figuras)
   save_tiff : bool    (si True, además guarda TIFF a 300 dpi)
   """
    # -------------------- datos --------------------
    X = df.drop('actual_construction_cost', axis=1).copy()
    y = df['actual_construction_cost'].copy()
 
    # Asegurar solo numéricas para evitar sorpresas con SHAP/corr
    X = X.select_dtypes(include=[np.number])
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
 
    # -------------------- modelo --------------------
    # Nota: 'reg:linear' está deprecado; usar 'reg:squarederror'
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
 
    # =========================================================
    # 1) Importancia por gain (XGBoost)
    # =========================================================
    gain = model.get_booster().get_score(importance_type='gain')
    imp_df = pd.DataFrame({'feature': list(gain.keys()), 'gain': list(gain.values())})
    if imp_df.empty:
        print("No se pudieron extraer importancias del booster (gain).")
    else:
        imp_df = imp_df.sort_values('gain', ascending=False)
        if top_n is not None:
            imp_df = imp_df.head(top_n)
 
        vals = imp_df['gain'].values
        labels_val = [f"{v:.2f}" for v in vals]
 
        fig, ax = plt.subplots(figsize=(12, max(4, 0.35*len(imp_df))))
        y_pos = range(len(imp_df))
 
        ax.barh(y_pos, vals, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp_df['feature'], fontsize=11)
        ax.invert_yaxis()
 
        ax.set_xlabel("Feature importance (gain)", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        ax.set_title(f"XGBoost Feature Importance (gain) — R²={r2:.3f}", fontsize=13, weight='bold')
 
        max_val = vals.max() if len(vals) else 1.0
        pad = 0.02 * max_val
        ax.set_xlim(0, max_val * 1.12)
 
        for i, v in enumerate(vals):
            inside = v > 0.90 * max_val
            if inside:
                ax.text(v - pad, i, labels_val[i], va='center', ha='right', fontsize=10, color='white')
            else:
                ax.text(v + pad, i, labels_val[i], va='center', ha='left', fontsize=10, color='black', clip_on=False)
 
        ax.xaxis.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
        plt.tight_layout()
        fname = f'feature_importance_external_{nature}.pdf'
        plt.savefig(fname, bbox_inches='tight')
        if save_tiff:
            plt.savefig(f'feature_importance_external_{nature}.tiff', dpi=300, bbox_inches='tight')
        plt.show()
 
    # =========================================================
    # 2) SHAP: explainer y valores
    # =========================================================
    # Para árboles, TreeExplainer es muy eficiente
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
 
    # En algunos entornos, SHAP usa pyplot global; forzamos estilo limpio
    plt.rcParams.update({'font.size': 11})
 
    # -------------------- SHAP summary (dot) --------------------
    plt.figure(figsize=(20, 6 + 0.15 * X_test.shape[1]))
    # Si se desea limitar a top_n, reducimos X_test y shap_values con esas columnas
    if top_n is not None:
        # orden global por |SHAP| medio
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-mean_abs)[:top_n]
        X_show = X_test.iloc[:, order]
        shap_show = shap_values[:, order]
    else:
        X_show = X_test
        shap_show = shap_values
 
    shap.summary_plot(
        shap_show,
        X_show,
        show=False  # para poder guardar
    )
    #plt.title(f"SHAP Summary (dot) — {nature}", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(f'shap_summary_dot_{nature}.pdf', bbox_inches='tight')
    if save_tiff:
        plt.savefig(f'shap_summary_dot_{nature}.tiff', dpi=300, bbox_inches='tight')
    plt.show()
 
    # -------------------- SHAP summary (bar) --------------------
    plt.figure(figsize=(8, max(4, 0.3 * X_show.shape[1])))
    shap.summary_plot(
        shap_show,
        X_show,
        plot_type='bar',
        show=False
    )
    plt.title(f"SHAP Global Importance (|mean SHAP|) — {nature}", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(f'shap_summary_bar_{nature}.pdf', bbox_inches='tight')
    if save_tiff:
        plt.savefig(f'shap_summary_bar_{nature}.tiff', dpi=300, bbox_inches='tight')
    plt.show()
 
    # -------------------- SHAP dependence (top feature) --------------------
    # Tomamos la feature con mayor |SHAP| medio
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs_shap))
    top_feat = X_test.columns[top_idx]
 
    plt.figure(figsize=(7,5))
    shap.dependence_plot(
        top_feat,
        shap_values,
        X_test,
        interaction_index='auto',  # deja que SHAP elija interacción informativa
        show=False
    )
    plt.title(f"SHAP Dependence — {top_feat}", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{nature}_{top_feat}.pdf', bbox_inches='tight')
    if save_tiff:
        plt.savefig(f'shap_dependence_{nature}_{top_feat}.tiff', dpi=300, bbox_inches='tight')
    plt.show()
        
    # shap_values es un arreglo [n_obs, n_features]
    # X_test tiene las columnas (features)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Construir dataframe ordenado
    shap_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    feature_important = model.get_booster().get_score(importance_type='gain')
    keys              = list(feature_important.keys())
    values            = list(feature_important.values())
    sorted_features   = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
   
    return shap_importance


def svm_regresion(X_train, y_train):
    modelo = make_pipeline(StandardScaler(), SVR(kernel='linear', C=1.0, epsilon=0.2))  # Ajusta los hiperparámetros si es necesario
    modelo.fit(X_train, y_train)
    return modelo

def ann_regresion(X_train, y_train):
    # Crear el modelo
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Entrenar el modelo
    modelo.fit(X_train, y_train, epochs=500, verbose=0)
    
    return modelo

def ann_regresion_2(X_train, y_train):
    breakpoint()
    # Split validación
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Modelo
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_tr.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7)

    # Entrenar
    history = modelo.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    return history
    
def rf_regresion(X_train, y_train):
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)  # Ajusta los hiperparámetros si es necesario
    modelo.fit(X_train, y_train)
    return modelo    

def modeling_regresion_db2(df_clustered, nature, ml_tech):
    
    df_subproj_ext = df_clustered.drop(['built_area',
    'lot_area',
    'total_prelim_cost_est',
    'prelim_cost_est_est',
    'equi_prelim_cost',
    'duration',
    'unit_price',
    'output', 'actual_sale_price'], axis=1).copy()
    
    sorted_feat_subproj = feature_importance(df_subproj_ext, nature) 
       
    # 2. Definir las variables de entrada y salida
    features_iniciales = ['built_area', 'lot_area', 'total_prelim_cost_est', 'prelim_cost_est_est', 
                          'equi_prelim_cost', 'duration', 'unit_price']
    features_adicionales = list(sorted_feat_subproj.feature)
        
    target = 'actual_construction_cost'
    
    resultados = []
    errores_relativos = []
    num_features_list = []  # Lista para almacenar el número de features para cada error

    # 5. Generar los plots en una matriz
    num_cols = 4  # Número de columnas en la matriz de subplots
    num_rows = int(np.ceil((len(features_adicionales)+1)/ num_cols))  # Número de filas
        
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols , 3.5 * num_rows))  # Ajusta figsize según sea necesario
    

    for i in range(0, len(features_adicionales)):        # Calcular la posición del subplot en la matriz
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]  # Manejar caso de una sola fila
                
        # 3. Crear y evaluar el modelo con diferentes combinaciones de variables
        features = features_iniciales + features_adicionales[:i+1]

        X = df_clustered[features].values
        y = df_clustered[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
                
        if ml_tech == 'ann':
            modelo = ann_regresion(X_train, y_train)
        
        if ml_tech == 'svm':
            modelo = svm_regresion(X_train, y_train)
            
        if ml_tech == 'rf':
            modelo = rf_regresion(X_train, y_train)
    
        y_pred = modelo.predict(X_test).flatten() 
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
                
        # Accuracy no se usa normalmente en regresión, pero se puede calcular como 1 - error relativo
        accuracy = 1 - np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) 
        errores = (y_test - y_pred.flatten()) / y_test * 100
        errores = errores.tolist()        
                
        resultados.append([len(features), features_adicionales[i], r2, accuracy, mae])
        #errores_relativos.append(errores)
        errores_relativos.extend(errores)
        num_features_list.extend([len(features)] * len(errores))  # Agregar el número de features correspondiente a cada error
        
        # Calcular la regresión lineal para la ecuación en el plot
        reg = LinearRegression().fit(y_test.reshape(-1, 1), y_pred)
        pendiente = reg.coef_[0]
        intercepto = reg.intercept_
        
        # Crear el plot
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Ideal')
        
        ax.set_title(f'Number of features: {len(features)}')
        ax.set_xlabel('Actual Construction Cost ('+r'$ \$ $)')
        ax.set_ylabel('Predicted Construction Cost ('+r'$ \$ $)')
        
        # Agregar la ecuación y R² al plot
        ecuacion = f'y = {pendiente:.4f}x + {intercepto:.0f}\nR² = {r2:.4f}'
        ax.text(0.05, 0.95, ecuacion, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    
    # Ocultar subplots vacíos si hay menos plots que espacios en la matriz
    if len(resultados) < num_rows * num_cols:
        for i in range(len(resultados), num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.axis('off')  # Ocultar el subplot
    
    plt.tight_layout()
    #plt.suptitle(nature+'_'+ml_tech)
    plt.savefig('predicted_cost_'+nature+'_'+ml_tech+'.pdf',bbox_inches='tight')
    plt.show()
        
    df_results = pd.DataFrame(resultados)
    df_results.columns = ['Num Features', 'Feature', 'R²', 'Accuracy', 'MAE']
        
    data = pd.DataFrame({'Number of Features': num_features_list, 'Relative error (%)': errores_relativos})
        
    plt.figure(figsize=(6, 6))
    sns.boxplot(x='Number of Features', y='Relative error (%)', data=data, width=0.95, palette='Set2')
    plt.title('Relative Errors by Number of Features')
    plt.xticks(rotation=45, ha='right') 
    plt.ylim([-100, 100])  # Ajusta los valores según tus necesidades
    plt.tight_layout()
    plt.savefig('error_relative'+nature+'_'+ml_tech+'.pdf',bbox_inches='tight')
    plt.show()
    
    return df_results, sorted_feat_subproj
