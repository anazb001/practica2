import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from termcolor import colored, cprint
import scipy.stats as ss
from sklearn.cluster import KMeans
import warnings


###-------------------------------- Funciones de la cátedra utilizadas ---------------------------------------
def duplicate_columns(frame):
    '''
    Lo que hace la función es, en forma de bucle, ir seleccionando columna por columna del DF que se le indique
    y comparar sus values con los de todas las demás columnas del DF. Si son exactamente iguales, añade dicha
    columna a una lista, para finalmente devolver la lista con los nombres de las columnas duplicadas.
    '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups

### -----------------------

def dame_variables_categoricas(dataset=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
        -Descripción: Función que recibe un dataset y devuelve una lista con los nombres de las 
        variables categóricas
        -Inputs: 
            -- dataset: Pandas dataframe que contiene los datos
        -Return:
            -- lista_variables_categoricas: lista con los nombres de las variables categóricas del
            dataset de entrada con menos de 100 valores diferentes
            -- 1: la ejecución es incorrecta
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    lista_variables_categoricas = []
    other = []
    for i in dataset.columns:
        if (dataset[i].dtype!=float) & (dataset[i].dtype!=int):
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 100:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)

    return lista_variables_categoricas, other

### ----

def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop(target,axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final



#####

def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))



###----------------------------------------- Funciones Propias -----------------------------------------------

def tipos_vars(df=None, show=True): # inspirada en una fuunción de un antiguo alumno
    '''
    ----------------------------------------------------------------------------------------------------------
    Función tipos_vars:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe como argumento un dataframe, analiza cada una de sus variables y muestra
        en pantalla el listado, categorizando a cada una como "categoric","bool" o "numeric". Para
        variables categóricas y booleanas se muestra el listado de categorías. Si son numéricas solo
        se informa el Rango y la Media de la variable.
        Devuelve 3 listas, cada una con los nombres de las variables pertenecientes a cada grupo ("bools", "categoric"
        y "numeric").
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- show: Argumento opcional, valor por defecto True. Si show es True, entonces se mostrará la
        información básica de con cada categoría. Si es False, la función solo devuelve las listas con
        los nombres de las variables según su categoría.
    - Return:
        -- list_bools: listado con el nombre de las variables booleanas encontradas
        -- list_cat: listado con el nombre de las variables categóricas encontradas
        -- list_num: listado con el nombre de las variables numéricas encontradas
    '''
    # Realizo una verificación por si no se introdujo ningún DF
    if df is None:
        print(u'No se ha especificado un DF para la función')
        return None
    
    # Genero listas vacías a rellenar con los nombres de las variables por categoría
    list_bools = []
    list_cat = []
    list_num = []
    
    # Analizo variables, completo las listas e imprimo la información de cada variable en caso de que el Show no se desactive
    for i in df.columns:
        if df[i].dropna(axis=0, how='all').nunique() <= 2:
            list_bools.append(i)
            if show:
                print(f"{i} {colored('(boolean)','blue')} :  {df[i].unique()}")
        elif df[i].dropna(axis=0, how='all').nunique() < 60:
            list_cat.append(i)
            if show:
                print(f"{i} {colored('(categoric)','red')} (\033[1mType\033[0m: {df[i].dtype}): {df[i].unique()}")
        else:
            list_num.append(i)
            if show:
                print(f"{i} {colored('(numeric)','green')} : \033[1mRange\033[0m = [{pd.to_numeric(df[i]).min():.2f} to {pd.to_numeric(df[i]).max():.2f}], \033[1mMean\033[0m = {pd.to_numeric(df[i]).mean():.2f}")
    
    # Finalmente devuelvo las listas con los nombres de las variables por cada categoría
    return list_bools,list_cat,list_num

def clas_var(df=None):
    if df is None:
        print(u'No se ha especificado un DF para la función')
        return None
    df = df.astype({col: 'category' for col in df.select_dtypes(include='object').columns})

    data_bool, data_cat, data_num = tipos_vars(df, show=False)

    variables_to_move = [
    'CNT_FAM_MEMBERS', 'CNT_CHILDREN', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_YEAR',
    'HOUR_APPR_PROCESS_START']

    data_cat = [var for var in data_cat if var not in variables_to_move]
    data_num.extend([var for var in variables_to_move if var in df.columns])

    data_cat.append('NAME_CONTRACT_TYPE')
    data_bool.remove('NAME_CONTRACT_TYPE')

    data_num.append('OWN_CAR_AGE')
    data_cat.remove('OWN_CAR_AGE')

    df[data_num] = df[data_num].astype(float)

    return df, data_bool, data_cat, data_num


#####


def custom_plot(df, col_name, is_cont, target):
    """
    Esta función permite visualizar la distribución de una variable y su distribución condicional al target. Además, para datos muy ásimetricos, se utiliza una escala
    logarítmica para mejorar la visualización. En casos en que una variable categórica tiene muchas categorías, lo que impide ver los nombres de las mismas, se cambia la distribución 
    de las imágenes y la rotación de las etiquetas, de nuevo para mejorar la visualización.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    col_name : str
        Column name to plot
    is_cont : bool
        Whether the variable is continuous
    target : str
        Target variable name
    """
    if is_cont:
        # Create subplot grid for continuous variables
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=90)
        
        # Left plot: Histogram with conditional log scale
        hist_data = df[col_name].value_counts(bins=30)
        if hist_data.max() > 150000:
            sns.histplot(df[col_name], bins=30, kde=False, log_scale=True, color='skyblue', ax=ax1)
        else:
            sns.histplot(df[col_name], bins=30, kde=False, color='skyblue', ax=ax1)
        ax1.set_title(f'Distribution of {col_name}')
        ax1.set_xlabel(col_name)
        ax1.set_ylabel('Count')
        
        # Right plot: Target-based barplot
        sns.boxplot(data=df, x=col_name, y=df[target].astype('string'), palette=['deepskyblue', 'crimson'], ax=ax2)
        ax2.set_ylabel('')  
        ax2.set_title(f'{col_name} by {target}')
        
    else:
        # Check number of unique categories
        n_unique = df[col_name].nunique()
        
        if n_unique < 6:
            # Create horizontal subplot grid for few categories
            f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
            rotation = 0
        else:
            # Create vertical subplot grid for many categories
            f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
            rotation = 90
            
        # Left/Top plot: Value counts
        value_counts = df[col_name].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax1, palette='YlGnBu')
        ax1.set_title(f'Distribution of {col_name}')
        ax1.set_xlabel(col_name)
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=rotation)
        
        # Right/Bottom plot: Target-based distribution
        target_props = df.groupby(col_name)[target].value_counts(normalize=True).unstack()
        target_props.plot(kind='bar', ax=ax2, color=['deepskyblue', 'crimson'])
        ax2.set_title(f'{col_name} Distribution by {target}')
        ax2.set_xlabel(col_name)
        ax2.set_ylabel('Proportion')
        ax2.tick_params(axis='x', rotation=rotation)
        ax2.legend(title=target)
    
    plt.tight_layout()


    ####


def corr_cat(df,target=None,target_transform=False): # obtenida del github de un antiguo alumno
    '''
    ----------------------------------------------------------------------------------------------------------
    Función corr_cat:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe como un dataframe, detecta las variables categóricas y calcula una especie de
        matriz de correlaciones mediante el uso del estadístico Cramers V. En la función se incluye la
        posibilidad de que se transforme a la variable target a string si no lo fuese y que se incluya en la
        lista de variables a analizar. Esto último  puede servir sobre todo para casos en los que la variable
        target es un booleano o está codificada.
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- target: String con nombre de la variable objetivo
        -- target_transform: Transforma la variable objetivo a string para el procesamiento y luego la vuelve
        a su tipo original.
    - Return:
        -- corr_cat: matriz con los Cramers V cruzados.
    '''
    df_cat_string = list(df.select_dtypes('category').columns.values)
    
    if target_transform:
        t_type = df[target].dtype
        df[target] = df[target].astype('string')
        df_cat_string.append(target)

    corr_cat = []
    vector = []

    for i in df_cat_string:
        vector = []
        for j in df_cat_string:
            confusion_matrix = pd.crosstab(df[i], df[j])
            vector.append(cramers_v(confusion_matrix.values))
        corr_cat.append(vector)

    corr_cat = pd.DataFrame(corr_cat, columns=df_cat_string, index=df_cat_string)
    
    if target_transform:
        df_cat_string.pop()
        df[target] = df[target].astype(t_type)

    return corr_cat

#####
    
def find_missing_value_mismatches(df, suffixes=['_AVG', '_MODE', '_MEDI']):
    """
    Find mismatches in missing value positions between variables with specified suffixes.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    suffixes (list): List of suffixes to compare.

    Returns:
    list: A list of tuples containing the variable pairs and the mismatch indices.
    """
    # Identify variables with the specified suffixes
    variables = df.columns

    # Find base variable names
    base_names = set()
    for var in variables:
        for suffix in suffixes:
            if var.endswith(suffix):
                base_names.add(var[:-len(suffix)])

    # Compare missing value positions and detect mismatches
    results = []
    for base in base_names:
        avg_col = base + '_AVG'
        mode_col = base + '_MODE'
        medi_col = base + '_MEDI'

        if avg_col in df.columns and mode_col in df.columns:
            mismatches = df[avg_col].isna() != df[mode_col].isna()
            if mismatches.any():
                results.append((avg_col, mode_col, mismatches))

        if avg_col in df.columns and medi_col in df.columns:
            mismatches = df[avg_col].isna() != df[medi_col].isna()
            if mismatches.any():
                results.append((avg_col, medi_col, mismatches))

        if mode_col in df.columns and medi_col in df.columns:
            mismatches = df[mode_col].isna() != df[medi_col].isna()
            if mismatches.any():
                results.append((mode_col, medi_col, mismatches))

    return results

def analyze_and_plot(df, prefixes, suffixes):
    """
    Calculate mean, standard deviation, and generate boxplots for specified prefixes and suffixes.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    prefixes (list): List of prefixes to consider.
    suffixes (list): List of suffixes to consider.
    """
    # Loop through each prefix and generate boxplots for the three suffixes together
    for prefix in prefixes:
        columns = [prefix + suffix for suffix in suffixes if prefix + suffix in df.columns]
        if columns:
            # Calculate mean and standard deviation for each column
            for column in columns:
                mean_value = df[column].mean()
                std_value = df[column].std()
                print(f"{column} - Mean: {mean_value:.2f}, Std Dev: {std_value:.2f}")

            # Generate boxplots for the three columns together
            plt.figure(figsize=(10, 5))
            df[columns].boxplot()
            plt.title(f'Boxplots of {prefix} Variables')
            plt.show()

            
def drop_mode_medi_columns(df, prefixes):
    """
    Drop all _MODE and _MEDI columns from the dataset for the specified prefixes.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    prefixes (list): List of prefixes to consider.

    Returns:
    pd.DataFrame: The modified DataFrame with _MODE and _MEDI columns dropped.
    """
    suffixes_to_drop = ['_MODE', '_MEDI']
    columns_to_drop = []

    # Identify columns to drop
    for prefix in prefixes:
        for suffix in suffixes_to_drop:
            column_name = prefix + suffix
            if column_name in df.columns:
                columns_to_drop.append(column_name)
    # Drop the identified columns
    df = df.drop(columns=columns_to_drop, axis=1)
    return df

def get_percent_null_values_target(pd_loan, list_var_continuous, target):

    pd_final = pd.DataFrame()
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum()>0:
            pd_concat_percent = pd.DataFrame(pd_loan[target][pd_loan[i].isnull()]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0], 
                                         pd_concat_percent.iloc[0]]
            pd_concat_percent = pd_concat_percent.drop('TARGET',axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
            pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum()/pd_loan.shape[0]
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

def plot_correlation_heatmap(corr, threshold=0.4, figsize=(14,12), fontsize=8):
    """
    Plot correlation heatmap with selective annotations.
    
    Parameters:
    -----------
    corr : pandas DataFrame
        Correlation matrix
    threshold : float, optional
        Minimum absolute correlation to display (default 0.4)
    figsize : tuple, optional
        Figure size (default (14,12))
    fontsize : int, optional
    """

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, 
                annot=True, 
                fmt='.1f', 
                cmap='icefire', 
                ax=ax, 
                vmin=-1, 
                vmax=1)
    
    for t in ax.texts:
        value = float(t.get_text())
        is_last_row = t in ax.texts[-len(corr.iloc[0]):len(list(ax.texts))]
        
        if ((abs(value) >= threshold) and (abs(value) < 1)) or is_last_row:
            t.set_fontsize(fontsize)
        else:
            t.set_text("")
    
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.title('Matriz de correlaciones', fontdict={'size':'20'})
    
    return fig, ax

def max_coef(d, features=''):
   
    return np.where(abs(d) == np.nanmax(abs(d.values)), features, '')

def k_means_search(df, clusters_max, figsize=(6, 6)):   # obtenida del github de un antiguo alumno
    """
    ----------------------------------------------------------------------------------------------------------
    Función k_means_search:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función que ejecuta el modelo no supervisado k-means sobre el DataFrame introducido tantas veces como
        la cantidad máxima de clusters que se quiera analizar y devuelve un gráfico que muestra la suma de
        los cuadrados de la distancia para cada cantidad de clusters.
    - Imputs:
        - df: DataFrame de Pandas sobre el que se ejecuta el K-Means
        - clusters_max: número máximo de clusters que se quiere analizar.
        - figsize: tupla con el tamaño deseado para la suma de ambos gráficos.
    """
    sse = []
    list_k = list(range(1, clusters_max+1))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(df)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=figsize)
    plt.plot(list_k, sse, '-o')
    plt.xlabel(f'Number of clusters {k}')
    plt.ylabel('Sum of squared distance')
    plt.show()


def reg_coefs(df, sel_model):
    # Create a DataFrame with predictors and their coefficients
    df_coeficientes = pd.DataFrame(
        {'predictor': df.columns,
         'coef': sel_model.estimator_.coef_.flatten()}
    )
    # Plot the coefficients
    fig, ax = plt.subplots(figsize=(16, 6)) 
    ax.stem(df_coeficientes.predictor, df_coeficientes.coef, markerfmt=' ')
    plt.xticks(
        rotation=90,  
        ha='center',  
        size=8  
    )
    plt.tight_layout(pad=3.0)  
    ax.set_xlabel('Variable', fontsize=12)
    ax.set_ylabel('Coeficientes', fontsize=12)
    ax.set_title('Coeficientes del modelo'+ sel_model, fontsize=14)
    plt.subplots_adjust(bottom=0.4)  
    plt.show()
        

def trat_vars(df='None'):
    if df is None:
        print(u'No se ha especificado un DF para la función')
        return None
    df = df.astype({col: 'category' for col in df.select_dtypes(include='object').columns})

    data_bool, data_cat, data_num = tipos_vars(df, show=False)

    variables_to_move = [
    'DEF_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_QRT']

    data_cat = [var for var in data_cat if var not in variables_to_move]
    data_num.extend([var for var in variables_to_move if var in df.columns])

    data_cat.append('NAME_CONTRACT_TYPE')
    data_bool.remove('NAME_CONTRACT_TYPE')

    return data_bool, data_cat, data_num


def y_pred_modelo_base(y_train, X_test):
    """
    Devuelve un array de numpy con las predicciones del modelo base para los datos otorgados.
    """

    value_max = y_train.value_counts(normalize=True).idxmax()
    size = len(X_test)
    y_pred_base = np.random.choice(
        [value_max, 1 - value_max],
        size=size,
        p=[1 - value_max, value_max]
    )
    return y_pred_base

def all_metrics(y_true, y_pred):
    
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, fbeta_score, f1_score,
        precision_score, recall_score, confusion_matrix
    )

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F2 Score": fbeta_score(y_true, y_pred, beta=2),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def plot_recall_precission(recall_precision, y_true, y_pred_proba):  #obtenida del github de un antiguo alumno
    """
    ----------------------------------------------------------------------------------------------------------
    Función plot_recall_precission:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función basada en un ejemplo de la cátedra, en la que se grafican diferentes métricas del modelo en
        base a los distintos threshold posibles para determinar el valor de la clase objetivo. Las métricas
        que se grafican son:
            - Precision
            - Recall
            - F2 Score
            - F1 Score
        Además, también se muestra en la leyenda el threshold óptimo para maximizar el F2 Score.
    - Imputs:
        - recall_precision: lista de listas en las que cada elemento representa un threshold con sus
        respectivas méticas dentro. Es decir que cada lista dentro de la lista padre contendrá 5 elementos:
        el threhold y las 4 métricas nombradas.
    """
    plt.figure(figsize=(15, 5))
    ax = sns.pointplot(x = [round(element[0],2) for element in recall_precision], y=[element[1] for element in recall_precision],
                     color="red", label='Recall', scale=1)
    ax = sns.pointplot(x = [round(element[0],2) for element in recall_precision], y=[element[2] for element in recall_precision],
                     color="blue", label='Precission')
    ax = sns.pointplot(x = [round(element[0],2) for element in recall_precision], y=[element[3] for element in recall_precision],
                     color="gold", label='F2 Score', lw=2)
    ax = sns.pointplot(x = [round(element[0],2) for element in recall_precision], y=[element[4] for element in recall_precision],
                     color="limegreen", label='F1 Score', lw=1)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f2_score = ((1+(2**2)) * precision * recall) / ((2**2) * precision + recall)
    ix = np.argmax(f2_score)
    
    ax.scatter((round(thresholds[ix],2)*100), f2_score[ix], s=100, marker='o', color='black', label=f'Best F2 (th={thresholds[ix]:.3f}, f2={f2_score[ix]:.3f})', zorder=2)
    ax.set_title('Recall & Precision VS Threshold', fontdict={'fontsize':20})
    ax.set_xlabel('threshold')
    ax.set_ylabel('probability')
    ax.legend()
    
    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if(i%5 == 0) or (i%5 ==1) or (i%5 == 2) or (i%5 == 3):
            labels[i] = '' # skip even labels
            ax.set_xticklabels(labels, rotation=45, fontdict={'size': 10})
    plt.show()


def plot_pr_curve(y_true, y_pred_proba, title='Precision-Recall Curve', f_score_beta=1, model_name='Model', figsize=(7,5)):

    precision, recall, thresholds = plot_recall_precission(y_true, y_pred_proba)

    f_score = ((1+(f_score_beta**2)) * precision * recall) / ((f_score_beta**2) * precision + recall)
    ix = np.argmax(f_score)
    auc_rp = auc(recall, precision)
    print(f'Best Threshold = {thresholds[ix]:.5f}, F{f_score_beta} Score = {f_score[ix]:.3f}, AUC = {auc_rp:.4f}')
    
    fig, ax = plt.subplots(figsize=figsize)
    no_skill= len(y_true[y_true==1])/len(y_true)
    ax.plot([0,1],[no_skill, no_skill], linestyle='--', label='No Skill', color='dodgerblue', lw=3)
    ax.plot(recall, precision, marker='.', label=model_name, color='orange')
    ax.scatter(recall[ix], precision[ix], s=100, marker='o', color='black', label=f'Best', zorder=2)
    ax.set_title(str(title), fontdict={'fontsize':18})
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(alpha=0.5)

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', figsize=(20, 6)): #función inspirada en otra de un antiguo alumno
  
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from matplotlib import rc

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Matriz de confusión absoluta
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, cmap='Blues', values_format=',.0f', ax=axes[0]
    )
    axes[0].set_title(f'{title}', fontdict={'fontsize': 18})
    axes[0].set_xlabel('Predicted Label', fontdict={'fontsize': 15})
    axes[0].set_ylabel('True Label', fontdict={'fontsize': 15})

    # Matriz de confusión normalizada
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, cmap='Blues', normalize='true', values_format='.2%', ax=axes[1]
    )
    axes[1].set_title(f'{title} - Normalized', fontdict={'fontsize': 18})
    axes[1].set_xlabel('Predicted Label', fontdict={'fontsize': 15})
    axes[1].set_ylabel('True Label', fontdict={'fontsize': 15})

    plt.tight_layout()
    plt.show()
