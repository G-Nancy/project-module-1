import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.feature_selection import VarianceThreshold
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import kurtosis, skew, pearsonr
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import xarray as xr
import warnings
from joblib import Parallel, delayed
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm # Import tqdm for progress bars
import copy # For deepcopying models in parallel loops

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers= [logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Crop list to process
crops = [
    'maize_major', 'maize_second',
    'rice_major', 'rice_second',
    'soybean',
    'wheat_winter', 'wheat_spring'
]

multi_var_files = {
    'merged_hadex.nc': ['DTR', 'ETR', 'FD', 'ID', 'PRCPTOT', 'R10mm', 'R20mm',
                        'Rx1day', 'Rx5day', 'SU', 'TNn', 'TNx', 'TR', 'TXn', 'TXx'],
    'merged_cru_data.nc': ['cld', 'dtr', 'frs', 'tmp', 'tmn', 'tmx', 'wet'],
    'merged_gleam.nc': ['E', 'Eb', 'Ec', 'Ei', 'Ep', 'Es', 'Et', 'Ew', 'H', 'SMrz', 'SMs', 'S']
}
single_var_patterns = {
    'aet_merged.nc': 'aet', 'def_merged.nc': 'def', 'pet_merged.nc': 'pet',
    'ppt_merged.nc': 'ppt', 'q_merged.nc': 'q', 'soil_merged.nc': 'soil',
    'srad_merged.nc': 'srad', 'swe_merged.nc': 'swe',
    'vap_merged.nc': 'vap', 'ws_merged.nc': 'ws',
    'pdsi_merged.nc': 'PDSI', 'GPP_1981-2015.nc': 'GPP', 'PKU_GIMMS_LAI_1981_2015.nc': 'lai',
    'PKU_GIMMS_NDVI_1981_2015.nc': 'ndvi'
}

# Define models as a function or ensure they are re-initialized for each parallel job
# to avoid multiprocessing issues with shared model states.
# We'll deepcopy them within the parallel helper functions.
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42, objective="reg:squarederror"),
    "LightGBM": lgb.LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
}

def rmsle(y_true, y_pred):
    """Calculates Root Mean Squared Logarithmic Error."""
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def corr_reg(y_true, y_pred):
    """Calculates Pearson correlation coefficient between true and predicted values."""
    if y_true.ndim > 1: y_true = y_true.squeeze()
    if y_pred.ndim > 1: y_pred = y_pred.squeeze()
    
    if len(y_true) < 2 or len(y_pred) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]

def aic(n, mse, p):
    """Calculates Akaike Information Criterion."""
    return n * np.log(mse) + 2 * p

def bic(n, mse, p):
    """Calculates Bayesian Information Criterion."""
    return n * np.log(mse) + p * np.log(n)
    
def identify_target_locations(yield_file_path_dir, calendar_file_path_dir, crop: str):
    """Identify target locations for a specific crop"""
    logger.info(f"Identifying target locations for {crop}...")
    try:
        yield_file = os.path.join(yield_file_path_dir, f"{crop}_yield_data.csv")
        calendar_file = os.path.join(calendar_file_path_dir, f"{crop}_growth_stages.csv")

        yield_data = pd.read_csv(yield_file)
        calendar_data = pd.read_csv(calendar_file)

        yield_locations = set(zip(yield_data['lat'], yield_data['lon']))
        calendar_locations = set(zip(calendar_data['lat'], calendar_data['lon']))
        common = yield_locations & calendar_locations

        logger.info(f"Found {len(common)} common locations for {crop}")
        return list(common)
    except Exception as e:
        logger.error(f"Error identifying locations for {crop}: {e}", exc_info=True)
        return []

def load_climate_data(data_path):
    """
    Loads climate data from NetCDF files specified in multi_var_files and single_var_patterns.
    """
    logger.info("Loading climate data...")
    climate_data = {}
    for file_path, variables in multi_var_files.items():
        try:
            full_path = os.path.join(data_path, file_path)
            if not os.path.exists(full_path):
                logger.warning(f"File not found: {full_path}")
                continue
            ds = xr.open_dataset(full_path, engine='netcdf4')
            logger.info(f"Available variables in {file_path}: {list(ds.data_vars.keys())}")
            for var in variables:
                if var in ds.data_vars:
                    try:
                        _ = ds[var].isel({dim: 0 for dim in ds[var].dims if dim != 'time'})
                        climate_data[var] = ds[var]
                        logger.info(f"Loaded {var} from {file_path}")
                    except Exception as var_error:
                        logger.warning(f"Skipping corrupted variable {var} from {file_path}: {var_error}")
                else:
                    logger.warning(f"Variable {var} not found in {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}", exc_info=True)
    for file_path, var_name in single_var_patterns.items():
        try:
            full_path = os.path.join(data_path, file_path)
            ds = xr.open_dataset(full_path, engine='netcdf4')
            data_vars = [v for v in ds.data_vars.keys() if v not in ['lat', 'lon']]
            if len(data_vars) > 0:
                actual_var = data_vars[0]
                climate_data[var_name] = ds[actual_var]
                logger.info(f"Loaded {var_name} ({actual_var}) from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}", exc_info=True)
    logger.info(f"Total climate variables loaded: {len(climate_data)}")
    return climate_data

def create_location_mapping(data_path, target_locations):
    """
    Creates a mapping from (lat, lon) pairs to the corresponding index in the climate data's 'points' dimension.
    Finds the closest climate data point for each target location.
    """
    ds = None
    for file_path in list(multi_var_files.keys()) + list(single_var_patterns.keys()):
        full_path = os.path.join(data_path, file_path)
        if os.path.isfile(full_path):
            try:
                ds = xr.open_dataset(full_path, engine='netcdf4')
                break
            except Exception as e:
                logger.warning(f"Could not open {full_path} for coordinate extraction: {e}")
    
    if ds is None:
        logger.error("Could not find any climate NetCDF file for coordinate extraction")
        return {}

    climate_lats = ds['lat'].values
    climate_lons = np.mod(ds['lon'].values, 360)
    location_mapping = {}
    for lat, lon in target_locations:
        norm_lon = np.mod(lon, 360)
        dists = np.sqrt((climate_lats - lat)**2 + (climate_lons - norm_lon)**2)
        idx = np.argmin(dists)
        if dists[idx] <= 1.0:
            location_mapping[(lat, lon)] = idx
    logger.info(f"Mapped {len(location_mapping)} out of {len(target_locations)} locations")
    return location_mapping

def _extract_single_variable_data_for_locations(var_name, data_array, location_mapping):
    """Helper function for parallel pre-extraction of a single variable across all mapped locations."""
    local_extracted_variable_data = {}
    for location, point_idx in location_mapping.items():
        try:
            if 'points' in data_array.dims:
                location_data = data_array.isel(points=point_idx).load() # Load data into memory
            else:
                continue # Skip if not 'points' dimension
            
            df = location_data.to_dataframe().reset_index()
            value_cols = [col for col in df.columns if col not in ['time', 'points']]
            if not value_cols:
                continue
            actual_var_name = value_cols[0]
            
            df['year'] = pd.to_datetime(df['time']).dt.year
            df['month'] = pd.to_datetime(df['time']).dt.month
            local_extracted_variable_data[location] = {var_name: df[['year', 'month', actual_var_name]]}
        except Exception as e:
            if "HDF error" in str(e) or "NetCDF" in str(e):
                logger.warning(f"Data corruption detected for {var_name} at location {location}: {e}")
            else:
                logger.error(f"Error extracting {var_name} for location {location}: {e}")
            continue
    return local_extracted_variable_data

def pre_extract_climate_data(climate_data, location_mapping, n_jobs=-1):
    """
    Pre-extracts time-series climate data for all mapped target locations in parallel.
    Parallelizes over climate variables.
    """
    logger.info("Pre-extracting climate data for all target locations in parallel...")
    
    if not climate_data or not location_mapping:
        logger.warning("No climate data or location mapping for pre-extraction.")
        return {}

    results_per_variable = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_extract_single_variable_data_for_locations)(var_name, data_array, location_mapping)
        for var_name, data_array in climate_data.items()
    )

    pre_extracted_data_final = {loc: {} for loc in location_mapping}
    for var_result_dict in results_per_variable:
        if var_result_dict:
            for location, var_data in var_result_dict.items():
                pre_extracted_data_final[location].update(var_data)
    
    pre_extracted_data_final = {loc: data for loc, data in pre_extracted_data_final.items() if data}

    logger.info(f"Pre-extracted data for {len(pre_extracted_data_final)} locations")
    return pre_extracted_data_final

def get_year_for_month(current_year, plant_month, harvest_month, target_month):
    """
    Determines the correct calendar year for a given target month within a growing season.
    `current_year` is the yield year.
    """
    if plant_month <= harvest_month:
        return current_year
    else:
        if target_month >= plant_month:
            return current_year - 1
        else:
            return current_year

def extract_features_from_pre_extracted_data(lat, lon, plant_month, mature_month, harvest_month, year, pre_extracted_data):
    """
    Extracts raw monthly climate data for the specific plant, mature, and harvest months.
    """
    features = {}
    if (lat, lon) not in pre_extracted_data:
        return features
    location_data = pre_extracted_data[(lat, lon)]

    key_months = {
        "plant": plant_month,
        "mid-stage": mature_month,
        "harvest": harvest_month
    }

    for stage_name, month_num in key_months.items():
        for var_name, df_full in location_data.items():
            value_col = df_full.columns[2]

            target_year_for_month = get_year_for_month(year, plant_month, harvest_month, month_num)
            
            monthly_data = df_full[(df_full['year'] == target_year_for_month) & (df_full['month'] == month_num)]
            
            if not monthly_data.empty:
                features[f"{stage_name}_month_{var_name}"] = monthly_data[value_col].iloc[0]
            else:
                features[f"{stage_name}_month_{var_name}"] = np.nan
    return features

def correlation_analysis(X, y, threshold=0.85):
    """
    Enhanced correlation analysis to identify and manage highly correlated features,
    and prioritize features based on their correlation with the target variable.
    """
    logger.info(f"Performing correlation analysis with threshold: {threshold}")
    
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    
    for column in upper_triangle.columns:
        high_corr_features = upper_triangle[column][upper_triangle[column] > threshold].index.tolist()
        for feature in high_corr_features:
            high_corr_pairs.append((column, feature, upper_triangle.loc[column, feature]))
    
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if high_corr_pairs:
        logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs:")
        for feat1, feat2, corr_val in high_corr_pairs[:10]:
            logger.info(f"    {feat1} - {feat2}: {corr_val:.3f}")
    
    to_drop = set()
    for feat1, feat2, _ in high_corr_pairs:
        feat1_data = X[feat1].dropna()
        feat2_data = X[feat2].dropna()
        
        y_aligned_feat1 = y.loc[feat1_data.index].dropna()
        y_aligned_feat2 = y.loc[feat2_data.index].dropna()

        corr_feat1_target = 0
        if len(feat1_data) > 1 and len(y_aligned_feat1) > 1 and np.std(feat1_data) > 0 and np.std(y_aligned_feat1) > 0:
            corr_feat1_target = abs(np.corrcoef(feat1_data, y_aligned_feat1)[0, 1])
        
        corr_feat2_target = 0
        if len(feat2_data) > 1 and len(y_aligned_feat2) > 1 and np.std(feat2_data) > 0 and np.std(y_aligned_feat2) > 0:
            corr_feat2_target = abs(np.corrcoef(feat2_data, y_aligned_feat2)[0, 1])

        if corr_feat1_target < corr_feat2_target:
            to_drop.add(feat1)
        else:
            to_drop.add(feat2)
    
    selected_features = [feature for feature in X.columns if feature not in to_drop]
    
    target_correlations = []
    for feature in selected_features:
        feature_data = X[feature].dropna()
        y_aligned = y.loc[feature_data.index].dropna()

        if feature_data.nunique() > 1 and len(feature_data) > 1 and len(y_aligned) > 1 and np.std(feature_data) > 0 and np.std(y_aligned) > 0:
            corr_with_target = abs(np.corrcoef(feature_data, y_aligned)[0, 1])
            target_correlations.append((feature, corr_with_target))
        else:
            logger.warning(f"Skipping correlation with target for '{feature}' due to low variance or insufficient non-NaN data.")
    
    target_correlations.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Top 10 features by correlation with target (after multicollinearity handling):")
    for feature, corr_val in target_correlations[:10]:
        logger.info(f"    {feature}: {corr_val:.3f}")
    
    return [feat for feat, _ in target_correlations]

def agglomerative_clustering(X, y, distance_threshold=0.5):
    """
    Performs agglomerative clustering on features and selects a representative from each cluster
    based on its correlation with the target variable.
    """
    X_clean = X.loc[:, X.notna().any(axis=0)]
    X_clean = X_clean.loc[:, X_clean.nunique() > 1]
    
    if X_clean.shape[1] < 2:
        logger.warning("Not enough features (less than 2) for clustering. Skipping agglomerative clustering step.")
        return []
        
    X_imputed = X_clean.fillna(X_clean.median())

    Z = linkage(X_imputed.T, method="ward")
    cluster_labels = fcluster(Z, t=distance_threshold, criterion="distance")
    logger.info(f"Cluster labels: {np.unique(cluster_labels)}")
    
    representative_features = []
    for label in np.unique(cluster_labels):
        if pd.isna(label):
            continue
        cluster_features = X_clean.columns[cluster_labels == label]
        if len(cluster_features) == 0:
            continue
        
        correlations = pd.Series(dtype=float)
        for col_feat in cluster_features:
            col_data = X_clean[col_feat].dropna()
            y_aligned = y.loc[col_data.index].dropna()
            if len(col_data) > 1 and len(y_aligned) > 1 and np.std(col_data) > 0 and np.std(y_aligned) > 0:
                correlations[col_feat] = abs(np.corrcoef(col_data, y_aligned)[0, 1])
            else:
                correlations[col_feat] = 0
        
        if not correlations.empty and correlations.max() > 0:
            best_feature = correlations.idxmax()
            representative_features.append(best_feature)
        else:
            logger.warning(f"No suitable representative found for cluster {label}.")
            
    return representative_features

def variance_threshold_selection(X, threshold=0.01):
    """
    Selects features based on their variance, removing features with variance below a threshold.
    """
    selector = VarianceThreshold(threshold=threshold)
    numeric_X = X.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if numeric_X.empty:
        logger.warning("No numeric features for variance threshold selection.")
        return []
        
    selector.fit(numeric_X)
    selected_features = numeric_X.columns[selector.get_support()]
    return selected_features.tolist()

def _calculate_single_model_importance(model_name, model_instance, X_clean, y_clean):
    """Helper function for parallel feature importance calculation for a single model."""
    try:
        cloned_model = copy.deepcopy(model_instance)
        cloned_model.fit(X_clean, y_clean)
        if hasattr(cloned_model, "feature_importances_"):
            return model_name, dict(zip(X_clean.columns, cloned_model.feature_importances_))
        elif hasattr(cloned_model, "coef_"):
            return model_name, dict(zip(X_clean.columns, np.abs(cloned_model.coef_)))
    except Exception as e:
        logger.error(f"Error calculating feature importance for {model_name}: {e}")
    return None, None

def feature_importance_analysis(models_dict, X, y, n_jobs=-1):
    """
    Calculates feature importances across multiple models in parallel.
    """
    feature_importances = {}
    
    X_clean = X.fillna(X.median())
    y_clean = y.fillna(y.median())

    if X_clean.empty or y_clean.empty:
        logger.warning(f"Skipping feature importance: X or y is empty after cleaning.")
        return []

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_calculate_single_model_importance)(name, model, X_clean, y_clean)
        for name, model in models_dict.items()
    )

    for model_name, importance_data in results:
        if model_name and importance_data:
            feature_importances[model_name] = importance_data
            
    if not feature_importances:
        logger.warning("No feature importances could be calculated from any model.")
        return []

    importance_df = pd.DataFrame(feature_importances).mean(axis=1).sort_values(ascending=False)
    return importance_df.head(15).index.tolist()

def causality_aware_feature_selection(X, y, correlation_threshold=0.1, p_value_threshold=0.05):
    """
    Filters features based on their statistical significance (p-value) and correlation with the target.
    """
    logger.info(f"Performing causality-aware feature selection (corr_thresh={correlation_threshold}, p_val_thresh={p_value_threshold})...")
    selected_causal_features =[]
    
    combined_data = pd.concat([X, y.rename('target_yield')], axis=1).dropna()
    
    if combined_data.empty:
        logger.warning("No data left for causality-aware feature selection after dropping NaNs.")
        return []

    X_clean = combined_data.drop('target_yield', axis=1)
    y_clean = combined_data['target_yield']

    for feature in X_clean.columns:
        if X_clean[feature].nunique() > 1 and len(X_clean[feature]) > 1:
            try:
                corr, p_value = pearsonr(X_clean[feature], y_clean)
                
                if abs(corr) >= correlation_threshold and p_value <= p_value_threshold:
                    selected_causal_features.append(feature)
            except Exception as e:
                logger.warning(f"Could not calculate correlation/p-value for feature '{feature}': {e}")
        else:
            logger.info(f"Skipping causality-aware check for '{feature}' due to low variance or insufficient data.")

    logger.info(f"Causality-aware feature selection identified {len(selected_causal_features)} features.")
    return selected_causal_features

def hybrid_feature_selection(X, y, threshold=0.01, n_jobs=-1):
    """
    Combines multiple feature selection techniques and then applies a causality-aware filter
    to identify a single set of key drivers.
    """
    print("Performing Enhanced Correlation Analysis...")
    corr_features = correlation_analysis(X, y, threshold=0.85)
    if corr_features and not X[corr_features].empty:
        corr_features = feature_importance_analysis(models, X[corr_features], y, n_jobs=n_jobs)
    else:
        corr_features = []
    
    print("Performing Agglomerative Clustering...")
    cluster_features = agglomerative_clustering(X, y)
    if cluster_features and not X[cluster_features].empty:
        cluster_features = feature_importance_analysis(models, X[cluster_features], y, n_jobs=n_jobs)
    else:
        cluster_features = []
    
    print("Performing Variance Threshold Selection...")
    vif_features = variance_threshold_selection(X, threshold=threshold)
    if vif_features and not X[vif_features].empty:
        vif_features = feature_importance_analysis(models, X[vif_features], y, n_jobs=n_jobs)
    else:
        vif_features = []
    
    initial_union_features = list(set(corr_features) | set(cluster_features) | set(vif_features))
    
    if initial_union_features and not X[initial_union_features].empty:
        final_key_drivers = causality_aware_feature_selection(X[initial_union_features], y,
                                                            correlation_threshold=0.1, p_value_threshold=0.05)
    else:
        final_key_drivers = []
        logger.warning("No features in initial union set for causality-aware filtering.")

    return final_key_drivers

def infer_causal_graph(X, y, key_drivers, crop_name, output_dir):
    """
    Infers a causal graph among key drivers and the target variable using the PC algorithm.
    Visualizes the graph and saves it.
    """
    logger.info(f"Inferring causal graph for {crop_name}...")
    
    data_for_causal_graph = pd.concat([X[key_drivers], y.rename('target_yield')], axis=1)
    data_for_causal_graph = data_for_causal_graph.dropna()
    
    if data_for_causal_graph.shape[0] < 50:
        logger.warning(f"Not enough data points ({data_for_causal_graph.shape[0]}) for robust causal graph inference for {crop_name}. Skipping.")
        return None, None
        
    if data_for_causal_graph.shape[1] < 2:
        logger.warning(f"Not enough features ({data_for_causal_graph.shape[1]}) for robust causal graph inference for {crop_name}. Skipping.")
        return None, None

    data_np = data_for_causal_graph.values
    node_names = list(data_for_causal_graph.columns)

    try:
        graph_model = pc(data_np, alpha=0.05, ci_test=fisherz, node_names=node_names)
        
        G = nx.DiGraph()
        for i, node_i in enumerate(node_names):
            G.add_node(node_i)
            for j, node_j in enumerate(node_names):
                if i == j:
                    continue
                if graph_model.graph[i, j] == 1:
                    G.add_edge(node_i, node_j)
                elif graph_model.graph[i, j] == 2:
                    G.add_edge(node_j, node_i)
                elif graph_model.graph[i, j] == 3:
                    G.add_edge(node_i, node_j)
                    G.add_edge(node_j, node_i)
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=0.8, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=3000, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=20, width=1.5, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        plt.title(f'Causal Graph for {crop_name} Key Drivers and Yield', fontsize=14)
        plt.axis('off')
        graph_path = os.path.join(output_dir, f"{crop_name}_causal_graph.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Causal graph saved to {graph_path}")
        
        return G, graph_model.graph
        
    except Exception as e:
        logger.error(f"Error inferring causal graph for {crop_name}: {e}")
        return None, None

def run_sensitivity_analysis(X, y, key_drivers, models_dict, crop_name, output_dir):
    """
    Performs sensitivity analysis using permutation importance for the identified key drivers.
    Note: `permutation_importance` uses `n_jobs` internally, so no direct `Parallel` call here.
    """
    logger.info(f"Performing sensitivity analysis (permutation importance) for {crop_name}...")
    
    if not key_drivers or X[key_drivers].empty or y.empty:
        logger.warning(f"Skipping sensitivity analysis for {crop_name}: no key drivers or empty data.")
        return {}

    X_reduced = X[key_drivers].fillna(X[key_drivers].median())
    y_clean = y.fillna(y.median())

    sensitivity_results = {}
    for model_name, model in models_dict.items():
        try:
            cloned_model = copy.deepcopy(model)
            cloned_model.fit(X_reduced, y_clean)
            
            result = permutation_importance(
                cloned_model, X_reduced, y_clean, n_repeats=10, random_state=42, n_jobs=-1
            )
            
            sorted_idx = result.importances_mean.argsort()[::-1]
            
            importance_df = pd.DataFrame({
                'Feature': X_reduced.columns[sorted_idx],
                'Importance_Mean': result.importances_mean[sorted_idx],
                'Importance_Std': result.importances_std[sorted_idx]
            })
            sensitivity_results[model_name] = importance_df
            
            logger.info(f"Permutation Importance for {model_name} ({crop_name}):\n{importance_df.head()}")
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis for {model_name} ({crop_name}): {e}")
            continue
            
    if sensitivity_results:
        all_importances = pd.concat([df.set_index('Feature')['Importance_Mean'] for df in sensitivity_results.values()], axis=1)
        avg_importance = all_importances.mean(axis=1).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 7))
        avg_importance.head(15).plot(kind='barh', color='darkgreen', alpha=0.7)
        plt.title(f'Average Permutation Importance for {crop_name} (Top 15)', fontsize=14)
        plt.xlabel('Mean Decrease in Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{crop_name}_permutation_importance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Permutation importance plot saved to {plot_path}")

    return sensitivity_results

def _calculate_partial_corr_for_feature(feature_to_corr, all_key_drivers, X_causal_data, y_causal_data):
    """Helper for parallel partial correlation calculation."""
    other_features = [f for f in all_key_drivers if f != feature_to_corr]
    
    try:
        if X_causal_data.shape[0] > len(other_features) + 1:
            if len(other_features) > 0:
                reg_y = LinearRegression().fit(X_causal_data[other_features], y_causal_data)
                y_residual = y_causal_data - reg_y.predict(X_causal_data[other_features])
                
                reg_x = LinearRegression().fit(X_causal_data[other_features], X_causal_data[feature_to_corr])
                x_residual = X_causal_data[feature_to_corr] - reg_x.predict(X_causal_data[other_features])
                
                if np.std(x_residual) > 0 and np.std(y_residual) > 0:
                    partial_corr, p_value = pearsonr(x_residual, y_residual)
                    return feature_to_corr, {'partial_correlation': partial_corr, 'p_value': p_value, 'significant': p_value < 0.05}
                else:
                    logger.warning(f"Insufficient variance in residuals for partial correlation for {feature_to_corr}.")
                    return feature_to_corr, None
            else: # If only one key driver, partial correlation is just simple correlation
                if np.std(X_causal_data[feature_to_corr]) > 0 and np.std(y_causal_data) > 0:
                    simple_corr, p_value = pearsonr(X_causal_data[feature_to_corr], y_causal_data)
                    return feature_to_corr, {'partial_correlation': simple_corr, 'p_value': p_value, 'significant': p_value < 0.05}
                else:
                    logger.warning(f"Insufficient variance for simple correlation for {feature_to_corr}.")
                    return feature_to_corr, None
        else:
            logger.warning(f"Not enough samples for partial correlation for {feature_to_corr}.")
            return feature_to_corr, None
    except Exception as e:
        logger.warning(f"Error calculating partial correlation for {feature_to_corr}: {e}")
        return feature_to_corr, None

def _analyze_single_causal_direction(feature_to_analyze, X_causal_data, y_causal_data):
    """Helper for parallel causal direction analysis."""
    from scipy.stats import kendalltau # Import locally to avoid pickling issues

    try:
        if len(X_causal_data) < 2:
            logger.warning(f"Not enough samples for causal direction analysis for {feature_to_analyze}.")
            return feature_to_analyze, None

        reg_xy = LinearRegression().fit(X_causal_data[[feature_to_analyze]], y_causal_data)
        residuals_xy = y_causal_data - reg_xy.predict(X_causal_data[[feature_to_analyze]])
        
        if np.std(X_causal_data[feature_to_analyze]) > 0 and np.std(residuals_xy) > 0:
            tau_xy, p_xy = kendalltau(X_causal_data[feature_to_analyze], residuals_xy)
        else:
            tau_xy, p_xy = np.nan, np.nan

        reg_yx = LinearRegression().fit(y_causal_data.values.reshape(-1, 1), X_causal_data[feature_to_analyze])
        residuals_yx = X_causal_data[feature_to_analyze] - reg_yx.predict(y_causal_data.values.reshape(-1, 1))

        if np.std(y_causal_data) > 0 and np.std(residuals_yx) > 0:
            tau_yx, p_yx = kendalltau(y_causal_data, residuals_yx)
        else:
            tau_yx, p_yx = np.nan, np.nan

        if not np.isnan(p_xy) and not np.isnan(p_yx):
            return feature_to_analyze, {
                'x_to_y_p_indep': p_xy,
                'y_to_x_p_indep': p_yx,
                'likely_direction': 'X->Y' if p_xy > p_yx else 'Y->X',
                'confidence': abs(p_xy - p_yx)
            }
        else:
            logger.warning(f"Skipping causal direction for {feature_to_analyze} due to NaN p-values.")
            return feature_to_analyze, None
    except Exception as e:
        logger.warning(f"Could not analyze causal direction for {feature_to_analyze}: {e}")
        return feature_to_analyze, None

def _analyze_single_interaction(feat1, feat2, X_causal_data, y_causal_data):
    """Helper for parallel feature interaction analysis."""
    try:
        if len(X_causal_data) > 2:
            interaction_data = X_causal_data[[feat1, feat2]].copy()
            interaction_data['interaction'] = X_causal_data[feat1] * X_causal_data[feat2]
            
            reg_base = LinearRegression().fit(X_causal_data[[feat1, feat2]], y_causal_data)
            reg_interaction = LinearRegression().fit(interaction_data, y_causal_data)
            
            r2_base = reg_base.score(X_causal_data[[feat1, feat2]], y_causal_data)
            r2_interaction = reg_interaction.score(interaction_data, y_causal_data)
            
            return f"{feat1}_x_{feat2}", {
                'r2_improvement': r2_interaction - r2_base,
                'interaction_coef': reg_interaction.coef_[-1] if len(reg_interaction.coef_) > 2 else np.nan,
                'significant': (r2_interaction - r2_base) > 0.01
            }
        else:
            logger.warning(f"Not enough samples for interaction analysis between {feat1} and {feat2}.")
            return f"{feat1}_x_{feat2}", None
    except Exception as e:
        logger.warning(f"Could not analyze interaction between {feat1} and {feat2}: {e}")
        return f"{feat1}_x_{feat2}", None

def causal_analysis(X, y, key_drivers, crop_name, output_dir, n_jobs=-1):
    """
    Perform causal analysis using key drivers, including partial correlation,
    causal direction analysis (Additive Noise Models), feature interaction analysis,
    and causal graph inference.
    """
    logger.info(f"Performing causal analysis for {crop_name}")
    
    X_causal = X[key_drivers].copy().dropna()
    y_causal = y.loc[X_causal.index].dropna()
    
    if X_causal.empty or y_causal.empty:
        logger.warning(f"Insufficient data for causal analysis after dropping NaNs for {crop_name}. Skipping.")
        return {}

    causal_results = {}
    
    logger.info("Computing partial correlations...")
    partial_correlations = {}
    if len(key_drivers) >= 1:
        partial_corr_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_calculate_partial_corr_for_feature)(feature, key_drivers, X_causal, y_causal)
            for feature in key_drivers
        )
        for feature, result in partial_corr_results:
            if result is not None: partial_correlations[feature] = result
    causal_results['partial_correlations'] = partial_correlations
                
    logger.info("Analyzing causal directions (simplified ANM approach)...")
    causal_directions = {}
    causal_direction_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_analyze_single_causal_direction)(feature, X_causal, y_causal)
        for feature in key_drivers
    )
    for feature, result in causal_direction_results:
        if result is not None: causal_directions[feature] = result
    causal_results['causal_directions'] = causal_directions

    logger.info("Analyzing feature interactions...")
    interaction_effects = {}
    if len(key_drivers) > 1:
        interaction_pairs = []
        for i, feat1 in enumerate(key_drivers):
            for feat2 in key_drivers[i+1:]:
                interaction_pairs.append((feat1, feat2))
        
        interaction_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_analyze_single_interaction)(feat1, feat2, X_causal, y_causal)
            for feat1, feat2 in interaction_pairs
        )
        for pair_name, result in interaction_results:
            if result is not None: interaction_effects[pair_name] = result
    causal_results['interaction_effects'] = interaction_effects
            
    causal_graph_obj, causal_graph_matrix = infer_causal_graph(X, y, key_drivers, crop_name, output_dir)
    causal_results['causal_graph_obj'] = causal_graph_obj
    causal_results['causal_graph_matrix'] = causal_graph_matrix
    
    print(f"\n=== CAUSAL ANALYSIS RESULTS FOR {crop_name.upper()} ===")
    
    print("\n1. Partial Correlations (controlling for other variables):")
    for feature, stats in partial_correlations.items():
        significance = "***" if stats['significant'] else ""
        print(f"    {feature}: {stats['partial_correlation']:.3f} (p={stats['p_value']:.3f}) {significance}")
    
    print("\n2. Causal Direction Analysis (Simplified ANM):")
    for feature, stats in causal_directions.items():
        print(f"    {feature}: {stats['likely_direction']} (confidence: {stats['confidence']:.3f})")
    
    print("\n3. Significant Feature Interactions (Top 5 R² improvement):")
    significant_interactions = {k: v for k, v in interaction_effects.items() if v['significant']}
    if significant_interactions:
        for interaction, stats in sorted(significant_interactions.items(),
                                         key=lambda item: item[1]['r2_improvement'], reverse=True)[:5]:
            print(f"    {interaction}: R² improvement = {stats['r2_improvement']:.3f}")
    else:
        print("    No significant interactions found.")

    if causal_graph_obj:
        print(f"\n4. Causal Graph Inferred. See '{crop_name}_causal_graph.png' for visualization.")
    else:
        print("\n4. Causal Graph inference skipped due to insufficient data or error.")
            
    return causal_results

def _validate_single_model(model_name, model_instance, X_train, X_test, y_train, y_test, n, p):
    """Helper function for parallel model validation."""
    try:
        cloned_model = copy.deepcopy(model_instance) # Deepcopy for isolated training
        cloned_model.fit(X_train, y_train)
        y_pred = cloned_model.predict(X_test)
        y_pred_clipped = np.maximum(0, y_pred)
        
        mse = mean_squared_error(y_test, y_pred)
        
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "RMSLE": rmsle(y_test, y_pred_clipped),
            "R2": r2_score(y_test, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred),
            "SMAPE": symmetric_mean_absolute_percentage_error(y_test, y_pred),
            "MedAE": median_absolute_error(y_test, y_pred),
            "EVS": explained_variance_score(y_test, y_pred),
            "Correlation": corr_reg(y_test, y_pred),
            "AIC": aic(n, mse, p),
            "BIC": bic(n, mse, p),
        }
        return model_name, metrics
    except Exception as e:
        logger.error(f"Error validating {model_name}: {e}")
    return None, None

def validate_key_drivers(key_drivers, X, y, models_dict, n_jobs=-1):
    """
    Validates the performance of models using the identified key drivers in parallel.
    """
    if not key_drivers or X[key_drivers].empty or y.empty:
        logger.warning("Skipping key driver validation: no key drivers or empty data.")
        return pd.DataFrame()

    X_reduced = X[key_drivers].fillna(X[key_drivers].median())
    y_clean = y.fillna(y.median())

    if len(X_reduced) < 2 or X_reduced.empty:
        logger.warning("Not enough samples or features in X_reduced for train-test split. Skipping validation.")
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_clean, test_size=0.2, random_state=42)
    
    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        logger.warning("Skipping key driver validation: not enough data after train-test split.")
        return pd.DataFrame()

    n = len(y_test)
    p = X_test.shape[1]
    
    results_list = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_validate_single_model)(model_name, model, X_train, X_test, y_train, y_test, n, p)
        for model_name, model in models_dict.items()
    )

    results = {}
    for model_name, metrics in results_list:
        if model_name and metrics:
            results[model_name] = metrics
            
    if not results:
        logger.warning("No validation results could be obtained from any model.")
        return pd.DataFrame()

    return pd.DataFrame(results).T

def plot_feature_importance(key_drivers_importance, crop_name, output_dir):
    """
    Plots bar chart of feature importances for the selected key drivers.
    """
    if key_drivers_importance.empty:
        logger.warning("No feature importance data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    
    key_drivers_importance = key_drivers_importance.sort_values(by="Importance", ascending=False)

    ax.barh(
        key_drivers_importance["Feature"],
        key_drivers_importance["Importance"],
        color="skyblue",
        edgecolor="black",
        height=0.6,
    )
    ax.set_xlabel("Normalized Importance", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_title(f"Feature Importance for Selected Key Drivers ({crop_name})", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{crop_name}_key_drivers_feature_importance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot for key drivers saved to {plot_path}")


def run_key_driver_analysis(X, y, crop_name, output_dir, n_jobs=-1):
    """
    Orchestrates the key driver identification, causal analysis, and validation steps for a single crop.
    Passes n_jobs for internal parallelization.
    """
    key_drivers = hybrid_feature_selection(X, y, threshold=0.01, n_jobs=n_jobs)
    print(f"Number of final key drivers selected: {len(key_drivers)}")

    if key_drivers and not X[key_drivers].empty:
        rf_model = RandomForestRegressor(random_state=42)
        X_for_importance = X[key_drivers].fillna(X[key_drivers].median())
        y_for_importance = y.fillna(y.median())

        if not X_for_importance.empty and not y_for_importance.empty:
            rf_model.fit(X_for_importance, y_for_importance)
            key_drivers_importance_df = pd.DataFrame({
                "Feature": X_for_importance.columns,
                "Importance": rf_model.feature_importances_
            })
            plot_feature_importance(key_drivers_importance_df, crop_name, output_dir)
        else:
            key_drivers_importance_df = pd.DataFrame(columns=["Feature", "Importance"])
            print("Insufficient data for key drivers importance plot.")
    else:
        key_drivers_importance_df = pd.DataFrame(columns=["Feature", "Importance"])
        print("Insufficient features for key drivers importance plot.")

    if key_drivers:
        causal_results = causal_analysis(X, y, key_drivers, crop_name, output_dir, n_jobs=n_jobs)
        plot_causal_analysis_results(causal_results, crop_name, output_dir)
    else:
        print("No key drivers available for causal analysis.")
    
    if key_drivers:
        run_sensitivity_analysis(X, y, key_drivers, models, crop_name, output_dir)
    else:
        print("No key drivers available for sensitivity analysis.")

    if key_drivers:
        results_key_drivers = validate_key_drivers(key_drivers, X, y, models, n_jobs=n_jobs)
        if not results_key_drivers.empty:
            print("\nResults with Selected Key Drivers:")
            print(results_key_drivers.round(3))
            results_key_drivers.to_csv(os.path.join(output_dir, f"{crop_name}_key_drivers_model_performance.csv"))
        else:
            print("\nValidation with Selected Key Drivers skipped due to insufficient data.")
    else:
        print("\nNo key drivers selected for validation.")


def process_single_crop(crop_name, data_path, output_dir, n_jobs_per_crop=-1):
    """
    Function to process a single crop, designed to be called in parallel.
    Now also passes n_jobs down to sub-functions for intra-crop parallelism.
    """
    logger.info(f"\n=== Processing crop: {crop_name} ===")
    yield_file_dir = 'yield_csvs'
    calendar_file_dir = 'gdhy_crop_stages_csv'

    target_locations = identify_target_locations(yield_file_dir, calendar_file_dir, crop_name)
    if not target_locations:
        logger.error(f"No common locations found for {crop_name}. Skipping.")
        return

    climate_data = load_climate_data(data_path)
    if not climate_data:
        logger.error(f"No climate data loaded for {crop_name}. Skipping.")
        return

    location_mapping = create_location_mapping(data_path, target_locations)
    if not location_mapping:
        logger.error(f"No locations mapped to climate data for {crop_name}. Skipping.")
        return

    pre_extracted_data = pre_extract_climate_data(climate_data, location_mapping, n_jobs=n_jobs_per_crop)
    if not pre_extracted_data:
        logger.error(f"No pre-extracted climate data for {crop_name}. Skipping.")
        return

    yield_data = pd.read_csv(os.path.join(yield_file_dir, f"{crop_name}_yield_data.csv"))
    planting_data = pd.read_csv(os.path.join(calendar_file_dir, f"{crop_name}_growth_stages.csv"))

    feature_list = []
    target_list = []

    for idx, row in yield_data.iterrows():
        lat, lon = row['lat'], row['lon']
        if (lat, lon) not in pre_extracted_data:
            continue
        planting_info = planting_data[(planting_data['lat'] == lat) & (planting_data['lon'] == lon)]
        if len(planting_info) == 0:
            continue
        plant_month = int(planting_info['plant'].iloc[0])
        mature_month = int(planting_info['mature'].iloc[0])
        harvest_month = int(planting_info['harvest'].iloc[0])

        for year in range(1981, 2016):
            yield_col = f'yield_{year}'
            if yield_col not in row.index or pd.isna(row[yield_col]):
                continue
            features = extract_features_from_pre_extracted_data(
                lat, lon, plant_month, mature_month, harvest_month, year, pre_extracted_data
            )
            if features:
                features['lat'] = lat
                features['lon'] = lon
                features['year'] = year
                feature_list.append(features)
                target_list.append(row[yield_col])
    
    feature_df = pd.DataFrame(feature_list)
    target_series = pd.Series(target_list)
    
    if len(feature_df) == 0:
        logger.error(f"No features extracted for crop: {crop_name}. Skipping further analysis.")
        return

    initial_cols = feature_df.columns.tolist()
    feature_df = feature_df.dropna(axis=1, thresh=len(feature_df) * 0.7)
    dropped_cols = set(initial_cols) - set(feature_df.columns)
    if dropped_cols:
        logger.info(f"Dropped {len(dropped_cols)} columns due to high missing values for {crop_name}.")

    for col in feature_df.columns:
        if feature_df[col].dtype in ['float64', 'int64']:
            if feature_df[col].isnull().any():
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
                logger.info(f"Imputed NaNs in column '{col}' with median for {crop_name}.")

    os.makedirs(output_dir, exist_ok=True)
    feature_df.to_csv(os.path.join(output_dir, f"{crop_name}_features.csv"), index=False)
    target_series.to_csv(os.path.join(output_dir, f"{crop_name}_targets.csv"), index=False)

    generate_descriptive_statistics_report(feature_df, target_series, crop_name, output_dir)

    run_key_driver_analysis(
        feature_df.drop(['lat', 'lon', 'year'], axis=1, errors='ignore'),
        target_series, crop_name, output_dir, n_jobs=n_jobs_per_crop
    )
    
    logger.info(f"Analysis complete for {crop_name}. Results saved in {output_dir}")


def run_for_all_crops(crop_list, data_path='test_gdhy_climate', output_dir='crop_results', n_jobs_outer=-1, n_jobs_inner=-1):
    """
    Runs the full key driver analysis pipeline for all specified crops in parallel, with progress tracking.
    n_jobs_outer: number of jobs for parallelizing across crops.
    n_jobs_inner: number of jobs for parallelizing within each crop's analysis steps.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting parallel processing for {len(crop_list)} crops with {n_jobs_outer} outer jobs.")
    
    # Use tqdm to wrap the delayed jobs for a progress bar
    Parallel(n_jobs=n_jobs_outer, prefer="processes")(
        delayed(process_single_crop)(crop_name, data_path, output_dir, n_jobs_per_crop=n_jobs_inner)
        for crop_name in tqdm(crop_list, desc="Processing Crops")
    )
    logger.info("All crop analyses complete.")

def generate_descriptive_statistics_report(feature_df, target_series, crop_name, output_dir):
    """
    Generate comprehensive descriptive statistics report for features and target.
    """
    logger.info(f"Generating descriptive statistics report for {crop_name}")
    
    numeric_features = feature_df.select_dtypes(include=[np.number])
    
    stats_dict = {}
    
    for column in numeric_features.columns:
        if column not in ['lat', 'lon', 'year']:
            data = numeric_features[column].dropna()
            if not data.empty:
                stats_dict[column] = {
                    'count': len(data),
                    'missing_count': len(numeric_features) - len(data),
                    'missing_percent': (len(numeric_features) - len(data)) / len(numeric_features) * 100 if len(numeric_features) > 0 else 0,
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'range': data.max() - data.min(),
                    'skewness': skew(data) if len(data) >= 4 else np.nan,
                    'kurtosis': kurtosis(data) if len(data) >= 4 else np.nan,
                    'cv': (data.std() / data.mean()) if data.mean() != 0 else np.nan
                }
            else:
                stats_dict[column] = {
                    'count': 0, 'missing_count': len(numeric_features), 'missing_percent': 100,
                    'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
                    'range': np.nan, 'skewness': np.nan, 'kurtosis': np.nan, 'cv': np.nan
                }
    
    target_data = target_series.dropna()
    if not target_data.empty:
        stats_dict['TARGET_YIELD'] = {
            'count': len(target_data),
            'missing_count': len(target_series) - len(target_data),
            'missing_percent': (len(target_series) - len(target_data)) / len(target_series) * 100 if len(target_series) > 0 else 0,
            'mean': target_data.mean(),
            'median': target_data.median(),
            'std': target_data.std(),
            'min': target_data.min(),
            'max': target_data.max(),
            'range': target_data.max() - target_data.min(),
            'skewness': skew(target_data) if len(target_data) >= 4 else np.nan,
            'kurtosis': kurtosis(target_data) if len(target_data) >= 4 else np.nan,
            'cv': (target_data.std() / target_data.mean()) if target_data.mean() != 0 else np.nan
        }
    else:
        stats_dict['TARGET_YIELD'] = {
            'count': 0, 'missing_count': len(target_series), 'missing_percent': 100,
            'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'range': np.nan, 'skewness': np.nan, 'kurtosis': np.nan, 'cv': np.nan
        }
    
    stats_df = pd.DataFrame(stats_dict).T
    
    for col in stats_df.columns:
        if col not in ['count', 'missing_count']:
            stats_df[col] = stats_df[col].round(4)
    
    stats_output_path = os.path.join(output_dir, f"{crop_name}_descriptive_statistics.csv")
    stats_df.to_csv(stats_output_path)
    
    print(f"\n=== DESCRIPTIVE STATISTICS SUMMARY FOR {crop_name.upper()} ===")
    print(f"Total features analyzed: {len(stats_df) - 1 if 'TARGET_YIELD' in stats_df.index else len(stats_df)}")
    print(f"Total observations: {len(feature_df)}")
    if 'TARGET_YIELD' in stats_df.index:
        print(f"Target variable (yield) statistics:")
        target_stats = stats_df.loc['TARGET_YIELD']
        print(f"    Mean: {target_stats['mean']:.2f}")
        print(f"    Median: {target_stats['median']:.2f}")
        print(f"    Std: {target_stats['std']:.2f}")
        print(f"    Skewness: {target_stats['skewness']:.2f}")
        print(f"    Kurtosis: {target_stats['kurtosis']:.2f}")
    
    high_cv_features = stats_df[stats_df['cv'] > 1].index.tolist()
    if high_cv_features:
        print(f"\nFeatures with high coefficient of variation (CV > 1):")
        for feature in high_cv_features[:10]:
            print(f"    {feature}: CV = {stats_df.loc[feature, 'cv']:.2f}")
    
    high_skew_features = stats_df[abs(stats_df['skewness']) > 2].index.tolist()
    if high_skew_features:
        print(f"\nFeatures with high skewness (|skewness| > 2):")
        for feature in high_skew_features[:10]:
            print(f"    {feature}: Skewness = {stats_df.loc[feature, 'skewness']:.2f}")
    
    high_missing_features = stats_df[stats_df['missing_percent'] > 10].index.tolist()
    if high_missing_features:
        print(f"\nFeatures with high missing values (>10%):")
        for feature in high_missing_features[:10]:
            print(f"    {feature}: {stats_df.loc[feature, 'missing_percent']:.1f}% missing")
    
    logger.info(f"Descriptive statistics saved to {stats_output_path}")
    
    return stats_df

def plot_causal_analysis_results(causal_results, crop_name, output_dir):
    """
    Create visualizations for causal analysis results (excluding the causal graph, which is separate).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Causal Analysis Results - {crop_name}', fontsize=16)
    
    if causal_results.get('partial_correlations'):
        partial_corr_data = causal_results['partial_correlations']
        valid_features = [f for f, stats in partial_corr_data.items() if not np.isnan(stats['partial_correlation'])]
        features = [f for f in valid_features if partial_corr_data[f]['partial_correlation']!= 0]
        correlations = [partial_corr_data[f]['partial_correlation'] for f in features]
        colors = ['red' if partial_corr_data[f]['significant'] else 'blue' for f in features]
        
        sorted_indices = np.argsort(correlations)
        features = [features[i] for i in sorted_indices]
        correlations = [correlations[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]

        if features:
            axes[0, 0].barh(features, correlations, color=colors, alpha=0.7)
            axes[0, 0].set_xlabel('Partial Correlation')
            axes[0, 0].set_title('Partial Correlations (Red = Significant)')
            axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        else:
            axes[0, 0].text(0.5, 0.5, 'No valid partial correlations to display.', horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Partial Correlations')
    else:
        axes[0, 0].text(0.5, 0.5, 'No partial correlations data.', horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Partial Correlations')

    if causal_results.get('causal_directions'):
        causal_dir_data = causal_results['causal_directions']
        features = list(causal_dir_data.keys())
        confidences = [causal_dir_data[f]['confidence'] for f in features]
        colors = ['green' if causal_dir_data[f]['likely_direction'] == 'X->Y' else 'orange' for f in features]
        
        sorted_indices = np.argsort(confidences)
        features = [features[i] for i in sorted_indices]
        confidences = [confidences[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]

        if features:
            axes[0, 1].barh(features, confidences, color=colors, alpha=0.7)
            axes[0, 1].set_xlabel('Causal Direction Confidence')
            axes[0, 1].set_title('Causal Direction Analysis (Green = X→Y, Orange = Y→X)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No valid causal directions to display.', horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Causal Direction Analysis')
    else:
        axes[0, 1].text(0.5, 0.5, 'No causal directions data.', horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Causal Direction Analysis')
    
    if causal_results.get('interaction_effects'):
        interaction_data = causal_results['interaction_effects']
        interactions = list(interaction_data.keys())
        improvements = [interaction_data[i]['r2_improvement'] for i in interactions]
        colors = ['purple' if interaction_data[i]['significant'] else 'gray' for i in interactions]
        
        top_interactions = sorted(zip(interactions, improvements, colors),
                                  key=lambda x: x[1], reverse=True)[:10]
        
        if top_interactions:
            int_names, int_values, int_colors = zip(*top_interactions)
            axes[1, 0].barh(int_names, int_values, color=int_colors, alpha=0.7)
            axes[1, 0].set_xlabel('R² Improvement')
            axes[1, 0].set_title('Feature Interactions (Purple = Significant)')
        else:
            axes[1, 0].text(0.5, 0.5, 'No significant interactions to display.', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Interactions')
    else:
        axes[1, 0].text(0.5, 0.5, 'No interaction effects data.', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Feature Interactions')
    
    axes[1, 1].axis('off')
    summary_text = f"""
    Causal Analysis Summary:
    
    • Partial Correlations: {len(causal_results.get('partial_correlations', {}))} features
    • Significant Partial Corr: {sum(1 for v in causal_results.get('partial_correlations', {}).values() if v.get('significant', False))}
    • Causal Directions: {len(causal_results.get('causal_directions', {}))} features
    • X→Y Directions: {sum(1 for v in causal_results.get('causal_directions', {}).values() if v.get('likely_direction') == 'X->Y')}
    • Interaction Effects: {len(causal_results.get('interaction_effects', {}))} pairs
    • Significant Interactions: {sum(1 for v in causal_results.get('interaction_effects', {}).values() if v.get('significant', False))}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{crop_name}_causal_analysis_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Causal analysis summary plot saved to {os.path.join(output_dir, f'{crop_name}_causal_analysis_summary.png')}")

if __name__ == "__main__":
    run_for_all_crops(
        crops,
        data_path='gdhy_climate',
        output_dir='test_crop_results_stage',
        n_jobs_outer=len(crops), # Parallelize across crops
        n_jobs_inner=-1 # Parallelize within each crop's analysis (e.g., model training, data extraction)
    )