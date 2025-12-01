#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:55:57 2024

@author: cristiantobar
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

def remove_outliers(df, df_encoded_ff):
    # OPTION 3: iqr filter: within 2.22 IQR (equiv. to z-score < 3)
    df_filtered          = df.copy()
    df_orig              = df_encoded_ff.copy()
    df_new               = df.drop('loan_interest_rate',axis= 1).copy()

    iqr                  = df_new.quantile(0.75, numeric_only=False) - df_new.quantile(0.25, numeric_only=False)
    lim                  = np.abs((df_new- df_new.median()) / iqr) < 2.22
    cols                 = df_new.select_dtypes('number').columns  # limits to a (float), b (int) and e (timedelta)
    df_orig.loc[:, cols] = df_new.where(lim, np.nan)
    df_orig.dropna(subset=cols, inplace=True) # drop rows with NaN in numerical columns
    df_filtered.loc[:, cols] = df_new.where(lim, np.nan)
    df_filtered.dropna(subset=cols, inplace=True)
    return df_orig, df_filtered

#def feature_importance(df):
def feature_importance(df, nature = "test", top_n=None, save_tiff=False):

    # X = df.drop('actual_construction_cost', axis=1).copy() # alternatively: X = df_no_missing.iloc[:,:-1]
    # y = df['actual_construction_cost'].copy()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # # Crear y entrenar el modelo
    # model    = xgb.XGBRegressor(objective="reg:linear", random_state=42)
    # model.fit(X_train, y_train)
    
    # y_predict = model.predict(X_test)
    # accu      = model.score(X_test, y_test)
    # r2        = r2_score(y_test, y_predict)
        
    # # Importancia de variables
    # importance = model.get_booster().get_score(importance_type='gain')
    # importance_df = pd.DataFrame({
    #     'Feature': list(importance.keys()),
    #     'Importance': list(importance.values())
    # }).sort_values(by="Importance", ascending=False)
    
    # # Gráfico mejorado
    # plt.figure(figsize=(12, 6))
    # sns.barplot(
    #     data=importance_df,
    #     x="Importance", y="Feature",
    #     palette="Blues_r"
    # )
    
    # # Añadir valores al final de las barras
    # for i, v in enumerate(importance_df["Importance"]):
    #     plt.text(v, i, f"{v:.2f}", va='center', ha='left', fontsize=10)
    
    # plt.xlabel("Gain-based Importance", fontsize=14)
    # plt.ylabel("Project Features", fontsize=14)
    # plt.title("Feature Importance in XGBoost Model", fontsize=16, weight="bold")
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.tight_layout()
    # plt.savefig("feature_importance_db2.pdf", bbox_inches="tight")
    # plt.show()
    
    # # #Obtener la importancia de variables y visualizar
    # # fig, ax  = plt.subplots(figsize=(15, 5))
    # # plot_importance(model, importance_type='gain', grid= False, values_format='{v:.2f}', ax=ax) 
    # # plt.savefig('feature_importance_db2.pdf',bbox_inches='tight')
    # # plt.show()    
    
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
    
    

    feature_important = model.get_booster().get_score(importance_type='gain')
    keys              = list(feature_important.keys())
    values            = list(feature_important.values())
    sorted_features   = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

    
    return sorted_features


def aglomerative_clustering_db2(df, currency, full_df):
    data_scaled = normalize(df)
    data_scaled = pd.DataFrame(data_scaled, columns=df.columns)
    data_scaled.head()
    
    plt.figure(figsize=(12, 6))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    plt.savefig('dendogram.pdf',bbox_inches='tight')
    plt.show()
    
    cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')  
    y_predict = cluster.fit_predict(data_scaled)
     
    # plt.scatter(df['built_area'], df['actual_construction_cost'], c=y_predict)
    # plt.xlabel('\n m2 \n')
    # plt.ylabel(currency)
    # plt.title("Built area vs. actual_construction_cost")  
    # plt.legend()
    # plt.show()
    
    # plt.scatter(df['lot_area'], df['actual_construction_cost'], c=y_predict)
    # plt.xlabel('\n m2  \n')
    # plt.ylabel(currency)
    # plt.title("lot_area vs. actual_construction_cost")  
    # plt.legend()
    # plt.show()
    
    # plt.scatter(df['total_prelim_cost_est'], df['actual_construction_cost'], c=y_predict)
    # plt.xlabel(currency)
    # plt.ylabel(currency)
    # plt.title("total_prelim_cost_est vs. actual_construction_cost")  
    # plt.legend()
    # plt.show()
    
    # plt.scatter(df['prelim_cost_est_est'], df['actual_construction_cost'], c=y_predict)
    # plt.xlabel(currency)
    # plt.ylabel(currency)
    # plt.title("prelim_cost_est_est vs. actual_construction_cost")  
    # plt.legend()
    # plt.show()
    
    # plt.scatter(df['equi_prelim_cost'], df['actual_construction_cost'], c=y_predict)
    # plt.xlabel(currency)
    # plt.ylabel(currency)
    # plt.title("equi_prelim_cost vs. actual_construction_cost")  
    # plt.legend()
    # plt.show()
    
    # plt.scatter(df['duration'], df['actual_construction_cost'], c=y_predict)
    # plt.xlabel('\n Weeks \n')
    # plt.ylabel(currency)    
    # plt.title("duration vs. actual_construction_cost")  
    # plt.legend()
    # plt.show()
    
    # plt.scatter(df['unit_price'], df['actual_construction_cost'], c=y_predict)
    # plt.xlabel(currency)
    # plt.ylabel(currency)
    # plt.title("unit_price vs. actual_construction_cost")  
    # plt.legend()
    # plt.show()
    
    y_predict = pd.DataFrame(y_predict)
    y_predict.columns = ['output'] #1 para descarado, 0 para no descarado
    
    df_temp = df.reset_index()
    df_temp_res = df_temp.drop(['index'], axis = 1).copy()
    
    new_df = pd.concat([df_temp_res, y_predict], axis = 1)
    
    full_df_temp = full_df.reset_index()
    full_df_temp_res = full_df_temp.drop(['index'], axis = 1).copy()
    new_full_df = pd.concat([full_df_temp_res, y_predict], axis = 1)
    
    X_encoded    = new_df.drop(['output'], axis = 1).copy()
    num_features = X_encoded.shape[1]
    Y            = new_df['output'].copy() 
    #Y_or         = df['weeks_delay'].copy()
    
    
    # plt.subplots(figsize=(7,2)) 
    # sns.boxplot(x=new_df['built_area'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('Built_area')
    # plt.xlabel(r'$m^{2}$')
    # plt.savefig('1_built_area.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,2)) 
    # sns.boxplot(x=new_df['lot_area'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('lot_area')
    # plt.xlabel(r'$m^{2}$')
    # plt.savefig('2_lot_area.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,2)) 
    # sns.boxplot(x=new_df['prelim_cost_est_est'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('prelim_cost_est_est')
    # plt.xlabel(r'$ \$ $')
    # plt.savefig('3_prelim_cost_est_est.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,2)) 
    # sns.boxplot(x=new_df['equi_prelim_cost'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('equi_prelim_cost')
    # plt.xlabel(r'$ \$ $')
    # plt.savefig('4_equi_prelim_cost.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,2)) 
    # sns.boxplot(x=new_df['total_prelim_cost_est'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('total_prelim_cost_est')
    # plt.xlabel(r'$ \$ $')
    # plt.savefig('5_total_prelim_cost_est.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,2)) 
    # sns.boxplot(x=new_df['duration'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('duration')
    # plt.xlabel('Weeks')
    # plt.savefig('6_duration.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,2)) 
    # sns.boxplot(x=new_df['unit_price'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('unit_price')
    # plt.xlabel(r'$ \$ $')
    # plt.savefig('7_unit_price.pdf',bbox_inches='tight')
    # plt.show()
    
    # plt.subplots(figsize=(7,2)) 
    # sns.boxplot(x=new_df['actual_construction_cost'], hue = pd.Categorical(new_df.output), width=.5, palette='Set2')
    # sns.set(font_scale=1.5)
    # plt.title('actual_construction_cost')
    # plt.xlabel(r'$ \$ $')
    # plt.savefig('8_actual_construction_cost.pdf',bbox_inches='tight')
    # plt.show()
    
    return new_df, new_full_df

def data_preparation_db2(df, currency, dollar2cop):
    
    df_encoded = pd.get_dummies(df, columns=[
                                            'loan_interest_rate'
                                            ])
        
    df['total_prelim_cost_est']        = df['total_prelim_cost_est']*10000000
    df['prelim_cost_est_est']          = df['prelim_cost_est_est']*10000
    df['equi_prelim_cost']             = df['equi_prelim_cost']*10000
    df['unit_price']                   = df['unit_price']*10000
    df['actual_sale_price']            = df['actual_sale_price']*10000
    df['actual_construction_cost']     = df['actual_construction_cost']*10000
    
        
    df['cumulative_liquidity']         = df['cumulative_liquidity']*10000000
    df['private_sector_investment']    = df['private_sector_investment']*10000000
    df['land_price_index']             = df['land_price_index']*10000000
    df['bank_loans_amou']              = df['bank_loans_amou']*10000000
    df['construc_cost_priv_time_fin']  = df['construc_cost_priv_time_fin']*10000
    df['construc_cost_priv_time_start']= df['construc_cost_priv_time_fin']*10000
    df['duration']                     = df['duration']*13

    if currency == 'Million COPm':
        #VALOR DEL DOLAR DE 10 DE NOVIEMBRE DE 2024
        df['total_prelim_cost_est']         = ((df['total_prelim_cost_est']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['prelim_cost_est_est']           = ((df['prelim_cost_est_est']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['equi_prelim_cost']              = ((df['equi_prelim_cost']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['unit_price']                    = ((df['unit_price']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['actual_sale_price']             = ((df['actual_sale_price']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['actual_construction_cost']      = ((df['actual_construction_cost']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        
        df['cumulative_liquidity']          = ((df['cumulative_liquidity']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['private_sector_investment']     = ((df['private_sector_investment']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['land_price_index']              = ((df['land_price_index']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['bank_loans_amou']               = ((df['bank_loans_amou']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['construc_cost_priv_time_fin']   = ((df['construc_cost_priv_time_fin']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['construc_cost_priv_time_start'] = ((df['construc_cost_priv_time_start']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['gold_price']                    = ((df['gold_price']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
    
    if currency == 'DOLLARm':
        df['total_prelim_cost_est']         = df['total_prelim_cost_est']/df['exchange_rate_to_dollar']
        df['prelim_cost_est_est']           = df['prelim_cost_est_est']/df['exchange_rate_to_dollar']
        df['equi_prelim_cost']              = df['equi_prelim_cost']/df['exchange_rate_to_dollar']
        df['unit_price']                    = df['unit_price']/df['exchange_rate_to_dollar']
        df['actual_sale_price']             = df['actual_sale_price']/df['exchange_rate_to_dollar']
        df['actual_construction_cost']      = df['actual_construction_cost']/df['exchange_rate_to_dollar']
        
        df['cumulative_liquidity']          = df['cumulative_liquidity']/df['exchange_rate_to_dollar']
        df['private_sector_investment']     = df['private_sector_investment']/df['exchange_rate_to_dollar']
        df['land_price_index']              = df['land_price_index']/df['exchange_rate_to_dollar']
        df['bank_loans_amou']               = df['bank_loans_amou']/df['exchange_rate_to_dollar']
        df['construc_cost_priv_time_fin']   = df['construc_cost_priv_time_fin']/df['exchange_rate_to_dollar']
        df['construc_cost_priv_time_start'] = df['construc_cost_priv_time_start']/df['exchange_rate_to_dollar']
        df['gold_price']                    = df['gold_price']/df['exchange_rate_to_dollar']
    
    
    descriptives = df.describe().round(2)

    
    df_wo_out_encoded, df_wo_out  = remove_outliers(df, df_encoded)
        
     #df_int = df_wo_outliers.iloc[:, [1,2,3,4,5,6,7,32]]
    df_int = df_wo_out[['built_area',
    'lot_area',
    'total_prelim_cost_est',
    'prelim_cost_est_est',
    'equi_prelim_cost',
    'duration',
    'unit_price',
    'actual_construction_cost']]
    # df_ext = df_wo_outliers.iloc[:,13:32]
    
    sorted_features = feature_importance(df_int)
    
    df_clustered, full_df_clustered    = aglomerative_clustering_db2(df_int, currency, df_wo_out)
    
    return df_wo_out_encoded, df_wo_out, sorted_features, df_clustered, full_df_clustered
    
    

