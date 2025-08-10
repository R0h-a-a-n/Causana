from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import numpy as np
import tempfile
import os
import csv
import math
import logging
from typing import List, Dict, Any, Optional
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def ensure_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Causana", version="2.0.0")

origins = [
    "http://localhost",
    "http://localhost:1234",
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataValidator:
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "stats": {}
        }
        
        if df.empty:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Dataframe is empty")
            return validation_result
        
        if len(df.columns) < 2:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Dataframe must have at least 2 columns for causality analysis")
            return validation_result
        
        numeric_columns = []
        date_columns = []
        text_columns = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            else:
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        date_columns.append(col)
                    except (ValueError, TypeError):
                        text_columns.append(col)
        
        validation_result["stats"]["rows"] = len(df)
        validation_result["stats"]["columns"] = len(df.columns)
        validation_result["stats"]["numeric_columns"] = len(numeric_columns)
        validation_result["stats"]["date_columns"] = len(date_columns)
        validation_result["stats"]["text_columns"] = len(text_columns)
        validation_result["stats"]["expected_numeric_after_preprocessing"] = len(numeric_columns) + len(date_columns)
        
        if len(df) < 50:
            validation_result["warnings"].append(f"Dataframe has only {len(df)} rows. At least 50 rows recommended for reliable analysis")
        
        if len(date_columns) > 0:
            validation_result["warnings"].append(f"Found {len(date_columns)} date column(s): {date_columns}. These will be converted to numeric values.")
        
        if len(text_columns) > 0:
            validation_result["warnings"].append(f"Found {len(text_columns)} text column(s): {text_columns}. These will be dropped during preprocessing.")
        
        missing_values = df.isnull().sum().sum()
        infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        validation_result["stats"]["missing_values"] = missing_values
        validation_result["stats"]["infinite_values"] = infinite_values
        
        if missing_values > 0:
            validation_result["warnings"].append(f"Found {missing_values} missing values. These will be filled during preprocessing.")
        
        if infinite_values > 0:
            validation_result["warnings"].append(f"Found {infinite_values} infinite values. These will be filled during preprocessing.")
        
        return validation_result

    @staticmethod
    def preprocess_data(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        df_processed = df.copy()
        
        date_columns = []
        for col in df_processed.columns:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                try:
                    pd.to_datetime(df_processed[col], errors='raise')
                    date_columns.append(col)
                except (ValueError, TypeError):
                    logger.warning(f"Dropping column with no numeric values: {col}")
                    df_processed = df_processed.drop(columns=[col])
        
        for col in date_columns:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                continue
            try:
                datetime_series = pd.to_datetime(df_processed[col])
                df_processed[col] = datetime_series.dt.year
                logger.info(f"Converted date column '{col}' to year values")
            except Exception as e:
                try:
                    datetime_series = pd.to_datetime(df_processed[col])
                    df_processed[col] = datetime_series.map(pd.Timestamp.toordinal)
                    logger.info(f"Converted date column '{col}' to ordinal values")
                except Exception as e2:
                    logger.warning(f"Dropping problematic date column '{col}': {e2}")
                    df_processed = df_processed.drop(columns=[col])
        
        df_processed = df_processed.fillna(df_processed.mean())
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        df_processed = df_processed.fillna(df_processed.mean())
        
        for col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(np.float64)
        
        if normalize:
            scaler = StandardScaler()
            df_processed = pd.DataFrame(
                scaler.fit_transform(df_processed),
                columns=df_processed.columns,
                index=df_processed.index
            )
        
        logger.info(f"Preprocessing complete. Shape: {df_processed.shape}, Columns: {df_processed.columns.tolist()}")
        logger.info(f"Data types: {df_processed.dtypes.tolist()}")
        
        return df_processed

class CausalityAnalyzer:
    
    @staticmethod
    def var_analysis(df: pd.DataFrame, lags: int = 5, alpha: float = 0.05, 
                    weight_metric: str = "neglogp") -> Dict[str, Any]:
        try:
            df_clean = df.copy()
            
            correlation_matrix = df_clean.corr()
            high_corr_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            if high_corr_pairs:
                logger.warning(f"Found highly correlated columns: {high_corr_pairs}")
                columns_to_remove = set()
                for col1, col2 in high_corr_pairs:
                    columns_to_remove.add(col2)
                
                df_clean = df_clean.drop(columns=list(columns_to_remove))
                logger.info(f"Removed {len(columns_to_remove)} highly correlated columns. New shape: {df_clean.shape}")
            
            if len(df_clean.columns) < 2:
                raise ValueError("After removing correlated columns, less than 2 columns remain")
            
            logger.info(f"VAR analysis input shape: {df_clean.shape}, dtypes: {df_clean.dtypes.tolist()}")
            logger.info(f"Sample data: {df_clean.head().values}")
            
            data_array = df_clean.values.astype(np.float64)
            logger.info(f"Numpy array shape: {data_array.shape}, dtype: {data_array.dtype}")
            
            df_final = pd.DataFrame(data_array, columns=df_clean.columns, dtype=np.float64)
            logger.info(f"Final DataFrame dtypes: {df_final.dtypes.tolist()}")
            
            np.random.seed(42)
            noise = np.random.normal(0, 1e-6, df_final.shape)
            df_final = df_final + noise
            logger.info("Added small noise to ensure positive definiteness")
            
            var = VAR(df_final)
            res = var.fit(maxlags=lags, ic="aic")
            k = int(res.k_ar)
            if k == 0:
                logger.warning("VAR model selected 0 lags, forcing 1 lag for causality testing")
                res = var.fit(maxlags=1, ic=None)
                k = 1
            
            nodes = df_final.columns.tolist()
            edges = []
            
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    if i != j:
                        try:
                            f_stat = res.test_causality(source, target, kind='f').statistic
                            p_value = res.test_causality(source, target, kind='f').pvalue
                            
                            if p_value < alpha:
                                if weight_metric == "neglogp":
                                    weight = -math.log10(p_value)
                                elif weight_metric == "fstat":
                                    weight = f_stat
                                elif weight_metric == "confidence":
                                    weight = 1 - p_value
                                else:
                                    weight = -math.log10(p_value)
                                
                                edges.append({
                                    "source": source,
                                    "target": target,
                                    "weight": weight,
                                    "p_value": p_value,
                                    "lag": k
                                })
                        except Exception as e:
                            logger.warning(f"Error testing causality from {source} to {target}: {e}")
                            continue
            
            logger.info(f"Analysis completed successfully for {len(df_final)} rows, {len(df_final.columns)} columns")
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"VAR analysis failed: {e}")
            raise ValueError(f"VAR analysis failed: {e}")

    @staticmethod
    def granger_causality(df: pd.DataFrame, max_lag: int = 5, alpha: float = 0.05) -> Dict[str, Any]:
        try:
            nodes = df.columns.tolist()
            edges = []
            
            for source in nodes:
                for target in nodes:
                    if source != target:
                        try:
                            data = df[[source, target]].dropna()
                            if len(data) < max_lag + 1:
                                continue
                            
                            result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                            
                            best_lag = None
                            best_p_value = 1.0
                            
                            for lag in range(1, max_lag + 1):
                                if lag in result:
                                    p_value = result[lag][0]['ssr_chi2test'][1]
                                    if p_value < best_p_value:
                                        best_p_value = p_value
                                        best_lag = lag
                            
                            if best_p_value < alpha and best_lag is not None:
                                weight = -math.log10(best_p_value)
                                edges.append({
                                    "source": source,
                                    "target": target,
                                    "weight": weight,
                                    "p_value": best_p_value,
                                    "lag": best_lag
                                })
                                
                        except Exception as e:
                            logger.warning(f"Error in Granger causality test from {source} to {target}: {e}")
                            continue
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Granger causality analysis failed: {e}")
            raise ValueError(f"Granger causality analysis failed: {e}")

    @staticmethod
    def correlation_analysis(df: pd.DataFrame, threshold: float = 0.7) -> Dict[str, Any]:
        try:
            correlation_matrix = df.corr()
            nodes = df.columns.tolist()
            edges = []
            
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    if i < j:
                        correlation = correlation_matrix.iloc[i, j]
                        if abs(correlation) >= threshold:
                            weight = abs(correlation)
                            edges.append({
                                "source": source,
                                "target": target,
                                "weight": weight,
                                "correlation": correlation
                            })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            raise ValueError(f"Correlation analysis failed: {e}")

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    method: str = Form("var"),
    lags: int = Form(5),
    alpha: float = Form(0.05),
    weight_metric: str = Form("neglogp"),
    normalize: bool = Form(True),
    max_lag: int = Form(5),
    correlation_threshold: float = Form(0.7)
):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            df = None
            delimiters = [',', ';', '\t', '|']
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(tmp_path, delimiter=delimiter)
                    logger.info(f"Successfully read CSV with delimiter: {repr(delimiter)}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to read with delimiter {repr(delimiter)}: {e}")
                    continue
            
            if df is None:
                try:
                    with open(tmp_path, "r") as f:
                        sample = f.read(2048)
                        detected_delimiter = csv.Sniffer().sniff(sample).delimiter
                    df = pd.read_csv(tmp_path, delimiter=detected_delimiter)
                    logger.info(f"Successfully read CSV with auto-detected delimiter: {repr(detected_delimiter)}")
                except Exception as e:
                    if "Could not determine delimiter" in str(e):
                        try:
                            df = pd.read_csv(tmp_path, delimiter=',')
                            logger.info("Successfully read CSV with comma delimiter as fallback")
                        except Exception as e2:
                            raise HTTPException(status_code=400, detail=f"Could not read CSV file. Please ensure it's a valid CSV with comma separators: {str(e2)}")
                    else:
                        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {str(e)}")
            
            validation_result = DataValidator.validate_dataframe(df)
            
            if not validation_result["is_valid"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Data validation failed",
                        "validation": validation_result
                    }
                )
            
            df_processed = DataValidator.preprocess_data(df, normalize)
            
            analysis_results = {}
            
            if method == "var" or method == "all":
                try:
                    var_result = CausalityAnalyzer.var_analysis(df_processed, lags, alpha, weight_metric)
                    analysis_results["var"] = var_result
                except Exception as e:
                    logger.error(f"VAR analysis failed: {e}")
                    if method == "var":
                        raise HTTPException(status_code=500, detail=f"VAR analysis failed: {e}")
            
            if method == "granger" or method == "all":
                try:
                    granger_result = CausalityAnalyzer.granger_causality(df_processed, max_lag, alpha)
                    analysis_results["granger"] = granger_result
                except Exception as e:
                    logger.error(f"Granger causality analysis failed: {e}")
                    if method == "granger":
                        raise HTTPException(status_code=500, detail=f"Granger causality analysis failed: {e}")
            
            if method == "correlation" or method == "all":
                try:
                    correlation_result = CausalityAnalyzer.correlation_analysis(df_processed, correlation_threshold)
                    analysis_results["correlation"] = correlation_result
                except Exception as e:
                    logger.error(f"Correlation analysis failed: {e}")
                    if method == "correlation":
                        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {e}")
            
            if not analysis_results:
                raise HTTPException(status_code=500, detail="No analysis methods completed successfully")
            
            return {
                "validation": validation_result,
                "analysis": analysis_results
            }
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "causana"}

@app.get("/methods")
async def get_available_methods():
    return {
        "methods": ["var", "granger", "correlation", "all"],
        "weight_metrics": ["neglogp", "fstat", "confidence"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
