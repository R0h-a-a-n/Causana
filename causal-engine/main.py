from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from enum import Enum
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import numpy as np
import tempfile
import os
import csv
import math
import logging
from typing import List, Dict, Any, Optional
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_ROWS = 30
    RECOMMENDED_ROWS = 50
    MAX_COLUMNS = 100
    CORRELATION_THRESHOLD = 0.95
    SMALL_NOISE = 1e-8

# Enums for type safety
class AnalysisMethod(str, Enum):
    VAR = "var"
    GRANGER = "granger"
    CORRELATION = "correlation"
    ALL = "all"

class WeightMetric(str, Enum):
    NEGLOGP = "neglogp"
    FSTAT = "fstat"
    CONFIDENCE = "confidence"

# Pydantic models for validation
class AnalysisRequest(BaseModel):
    method: AnalysisMethod = AnalysisMethod.VAR
    lags: int = Field(5, ge=1, le=20)
    alpha: float = Field(0.05, gt=0, lt=1)
    weight_metric: WeightMetric = WeightMetric.NEGLOGP
    normalize: bool = True
    max_lag: int = Field(5, ge=1, le=20)
    correlation_threshold: float = Field(0.7, ge=0, le=1)
    check_stationarity: bool = True

def ensure_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Causana",
    version="2.1.0",
    description="Causality Analysis API for Time Series Data"
)

# CORS configuration
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
    """Validates and preprocesses data for causality analysis."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataframe validation."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "stats": {}
        }
        
        # Check if empty
        if df.empty:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Dataframe is empty")
            return validation_result
        
        # Check minimum columns
        if len(df.columns) < 2:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Dataframe must have at least 2 columns for causality analysis")
            return validation_result
        
        # Check maximum columns
        if len(df.columns) > Config.MAX_COLUMNS:
            validation_result["warnings"].append(
                f"Dataframe has {len(df.columns)} columns. Analysis may be slow with > {Config.MAX_COLUMNS} columns"
            )
        
        # Categorize columns
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
        
        # Store statistics
        validation_result["stats"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(numeric_columns),
            "date_columns": len(date_columns),
            "text_columns": len(text_columns),
            "expected_numeric_after_preprocessing": len(numeric_columns) + len(date_columns)
        }
        
        # Check minimum rows
        if len(df) < Config.MIN_ROWS:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Dataframe has only {len(df)} rows. At least {Config.MIN_ROWS} rows required"
            )
            return validation_result
        
        if len(df) < Config.RECOMMENDED_ROWS:
            validation_result["warnings"].append(
                f"Dataframe has only {len(df)} rows. At least {Config.RECOMMENDED_ROWS} rows recommended for reliable analysis"
            )
        
        # Warn about non-numeric columns
        if date_columns:
            validation_result["warnings"].append(
                f"Found {len(date_columns)} date column(s): {date_columns[:5]}. These will be converted to numeric values."
            )
        
        if text_columns:
            validation_result["warnings"].append(
                f"Found {len(text_columns)} text column(s): {text_columns[:5]}. These will be dropped during preprocessing."
            )
        
        # Check for missing and infinite values
        missing_values = df.isnull().sum().sum()
        infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        validation_result["stats"]["missing_values"] = int(missing_values)
        validation_result["stats"]["infinite_values"] = int(infinite_values)
        
        if missing_values > 0:
            pct_missing = (missing_values / (df.shape[0] * df.shape[1])) * 100
            validation_result["warnings"].append(
                f"Found {missing_values} missing values ({pct_missing:.2f}%). These will be imputed."
            )
        
        if infinite_values > 0:
            validation_result["warnings"].append(
                f"Found {infinite_values} infinite values. These will be replaced."
            )
        
        # Check for constant columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                validation_result["warnings"].append(
                    f"Column '{col}' has constant or near-constant values and will be dropped"
                )
        
        return validation_result

    @staticmethod
    def check_stationarity(series: pd.Series, name: str) -> Dict[str, Any]:
        """Check if time series is stationary using ADF test."""
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            return {
                "variable": name,
                "is_stationary": result[1] < 0.05,
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "critical_values": {k: float(v) for k, v in result[4].items()}
            }
        except Exception as e:
            logger.warning(f"Stationarity test failed for {name}: {e}")
            return {
                "variable": name,
                "is_stationary": None,
                "error": str(e)
            }

    @staticmethod
    def preprocess_data(df: pd.DataFrame, normalize: bool = True, 
                       check_stationarity: bool = False) -> tuple:
        """Preprocess data with improved handling."""
        df_processed = df.copy()
        preprocessing_info = {
            "dropped_columns": [],
            "converted_columns": [],
            "constant_columns": [],
            "stationarity_tests": []
        }
        
        # Remove duplicate columns
        df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]
        
        # Handle date columns
        date_columns = []
        for col in df_processed.columns:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                try:
                    pd.to_datetime(df_processed[col], errors='raise')
                    date_columns.append(col)
                except (ValueError, TypeError):
                    pass
        
        for col in date_columns:
            try:
                datetime_series = pd.to_datetime(df_processed[col])
                # Convert to Unix timestamp for better handling
                df_processed[col] = datetime_series.astype(np.int64) / 10**9
                preprocessing_info["converted_columns"].append({
                    "column": col,
                    "from": "date",
                    "to": "unix_timestamp"
                })
                logger.info(f"Converted date column '{col}' to Unix timestamp")
            except Exception as e:
                logger.warning(f"Dropping problematic date column '{col}': {e}")
                df_processed = df_processed.drop(columns=[col])
                preprocessing_info["dropped_columns"].append(col)
        
        # Drop non-numeric columns
        non_numeric_cols = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            df_processed = df_processed.drop(columns=non_numeric_cols)
            preprocessing_info["dropped_columns"].extend(non_numeric_cols)
            logger.info(f"Dropped non-numeric columns: {non_numeric_cols}")
        
        # Remove constant columns
        for col in df_processed.columns:
            if df_processed[col].nunique() <= 1:
                df_processed = df_processed.drop(columns=[col])
                preprocessing_info["constant_columns"].append(col)
                logger.info(f"Dropped constant column: {col}")
        
        # Handle missing values with forward/backward fill first, then mean
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        df_processed = df_processed.fillna(df_processed.mean())
        
        # Handle infinite values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        df_processed = df_processed.fillna(df_processed.mean())
        
        # Ensure float64 dtype
        df_processed = df_processed.astype(np.float64)
        
        # Check stationarity if requested
        if check_stationarity:
            for col in df_processed.columns:
                stationarity_result = DataValidator.check_stationarity(df_processed[col], col)
                preprocessing_info["stationarity_tests"].append(stationarity_result)
        
        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            df_processed = pd.DataFrame(
                scaler.fit_transform(df_processed),
                columns=df_processed.columns,
                index=df_processed.index
            )
            logger.info("Data normalized using StandardScaler")
        
        logger.info(f"Preprocessing complete. Shape: {df_processed.shape}")
        
        return df_processed, preprocessing_info

class CausalityAnalyzer:
    """Performs various causality analyses."""
    
    @staticmethod
    def remove_multicollinearity(df: pd.DataFrame, threshold: float = Config.CORRELATION_THRESHOLD) -> tuple:
        """Remove highly correlated columns to prevent multicollinearity."""
        correlation_matrix = df.corr().abs()
        removed_columns = []
        
        # Get upper triangle of correlation matrix
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find columns with correlation above threshold
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            df_clean = df.drop(columns=to_drop)
            removed_columns = to_drop
            logger.info(f"Removed {len(to_drop)} highly correlated columns: {to_drop}")
        else:
            df_clean = df.copy()
        
        return df_clean, removed_columns

    @staticmethod
    def var_analysis(df: pd.DataFrame, lags: int = 5, alpha: float = 0.05, 
                    weight_metric: str = "neglogp") -> Dict[str, Any]:
        """Improved VAR analysis with better error handling."""
        try:
            # Remove multicollinearity
            df_clean, removed_cols = CausalityAnalyzer.remove_multicollinearity(df)
            
            if len(df_clean.columns) < 2:
                raise ValueError("After removing correlated columns, less than 2 columns remain")
            
            # Add small noise for numerical stability
            np.random.seed(42)
            noise = np.random.normal(0, Config.SMALL_NOISE, df_clean.shape)
            df_clean = df_clean + noise
            
            # Fit VAR model
            var = VAR(df_clean)
            
            # Try different lag selection methods
            try:
                res = var.fit(maxlags=lags, ic="aic")
                k = int(res.k_ar)
            except:
                logger.warning("AIC-based lag selection failed, using BIC")
                try:
                    res = var.fit(maxlags=lags, ic="bic")
                    k = int(res.k_ar)
                except:
                    logger.warning("BIC-based lag selection failed, using fixed lags")
                    res = var.fit(maxlags=min(lags, 3), ic=None)
                    k = min(lags, 3)
            
            if k == 0:
                logger.warning("VAR model selected 0 lags, forcing 1 lag")
                res = var.fit(maxlags=1, ic=None)
                k = 1
            
            nodes = df_clean.columns.tolist()
            edges = []
            
            # Test causality for each pair
            for source in nodes:
                for target in nodes:
                    if source != target:
                        try:
                            test_result = res.test_causality(source, target, kind='f')
                            f_stat = test_result.statistic
                            p_value = test_result.pvalue
                            
                            if p_value < alpha:
                                # Calculate weight based on metric
                                if weight_metric == "neglogp":
                                    weight = -math.log10(max(p_value, 1e-10))
                                elif weight_metric == "fstat":
                                    weight = float(f_stat)
                                elif weight_metric == "confidence":
                                    weight = 1 - p_value
                                else:
                                    weight = -math.log10(max(p_value, 1e-10))
                                
                                edges.append({
                                    "source": source,
                                    "target": target,
                                    "weight": float(weight),
                                    "p_value": float(p_value),
                                    "f_statistic": float(f_stat),
                                    "lag": k
                                })
                        except Exception as e:
                            logger.debug(f"Causality test failed for {source} -> {target}: {e}")
            
            return {
                "nodes": nodes,
                "edges": edges,
                "optimal_lag": k,
                "removed_columns": removed_cols,
                "model_aic": float(res.aic) if hasattr(res, 'aic') else None,
                "model_bic": float(res.bic) if hasattr(res, 'bic') else None
            }
            
        except Exception as e:
            logger.error(f"VAR analysis failed: {e}")
            raise ValueError(f"VAR analysis failed: {str(e)}")

    @staticmethod
    def granger_causality(df: pd.DataFrame, max_lag: int = 5, alpha: float = 0.05) -> Dict[str, Any]:
        """Improved Granger causality with parallel processing."""
        nodes = df.columns.tolist()
        edges = []
        failed_tests = []
        
        def test_pair(source, target):
            if source == target:
                return None
            try:
                data = df[[source, target]].dropna()
                if len(data) < max_lag + 10:  # Need enough data points
                    return None
                
                result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
                # Find best lag
                best_lag = None
                best_p_value = 1.0
                best_f_stat = 0.0
                
                for lag in range(1, max_lag + 1):
                    if lag in result:
                        # Use multiple tests for robustness
                        p_ssr = result[lag][0]['ssr_chi2test'][1]
                        p_f = result[lag][0]['ssr_ftest'][1]
                        f_stat = result[lag][0]['ssr_ftest'][0]
                        
                        # Use the more conservative p-value
                        p_value = max(p_ssr, p_f)
                        
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_lag = lag
                            best_f_stat = f_stat
                
                if best_p_value < alpha and best_lag is not None:
                    return {
                        "source": source,
                        "target": target,
                        "weight": float(-math.log10(max(best_p_value, 1e-10))),
                        "p_value": float(best_p_value),
                        "f_statistic": float(best_f_stat),
                        "lag": best_lag
                    }
                return None
                
            except Exception as e:
                logger.debug(f"Granger test failed for {source} -> {target}: {e}")
                return None
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for source in nodes:
                for target in nodes:
                    futures.append(executor.submit(test_pair, source, target))
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    edges.append(result)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "max_lag": max_lag
        }

    @staticmethod
    def correlation_analysis(df: pd.DataFrame, threshold: float = 0.7) -> Dict[str, Any]:
        """Enhanced correlation analysis."""
        correlation_matrix = df.corr()
        nodes = df.columns.tolist()
        edges = []
        
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i < j:
                    correlation = correlation_matrix.iloc[i, j]
                    if abs(correlation) >= threshold:
                        edges.append({
                            "source": source,
                            "target": target,
                            "weight": float(abs(correlation)),
                            "correlation": float(correlation),
                            "type": "positive" if correlation > 0 else "negative"
                        })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "correlation_matrix": correlation_matrix.to_dict()
        }

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    method: str = Form("var"),
    lags: int = Form(5),
    alpha: float = Form(0.05),
    weight_metric: str = Form("neglogp"),
    normalize: bool = Form(True),
    max_lag: int = Form(5),
    correlation_threshold: float = Form(0.7),
    check_stationarity: bool = Form(False)
):
    """Main analysis endpoint with comprehensive error handling."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="File must be a CSV. Supported format: .csv"
            )
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {Config.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Try reading with different delimiters
            df = None
            delimiters = [',', ';', '\t', '|']
            
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(tmp_path, delimiter=delimiter, encoding='utf-8')
                    if len(df.columns) > 1:  # Valid dataframe
                        logger.info(f"Successfully read CSV with delimiter: {repr(delimiter)}")
                        break
                except:
                    continue
            
            # Try with different encodings if UTF-8 fails
            if df is None or len(df.columns) <= 1:
                for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(tmp_path, encoding=encoding)
                        if len(df.columns) > 1:
                            logger.info(f"Successfully read CSV with encoding: {encoding}")
                            break
                    except:
                        continue
            
            if df is None or len(df.columns) <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="Could not parse CSV file. Please ensure it's a valid CSV with proper delimiters."
                )
            
            # Validate parameters
            if not (1 <= lags <= 20):
                raise HTTPException(status_code=400, detail="Lags must be between 1 and 20")
            if not (0 < alpha < 1):
                raise HTTPException(status_code=400, detail="Alpha must be between 0 and 1")
            
            # Validate dataframe
            validation_result = DataValidator.validate_dataframe(df)
            
            if not validation_result["is_valid"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Data validation failed",
                        "validation": ensure_json_serializable(validation_result)
                    }
                )
            
            # Preprocess data
            df_processed, preprocessing_info = DataValidator.preprocess_data(
                df, normalize, check_stationarity
            )
            
            # Check if enough columns remain
            if len(df_processed.columns) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="After preprocessing, less than 2 numeric columns remain. Cannot perform causality analysis."
                )
            
            # Perform analyses
            analysis_results = {}
            errors = {}
            
            if method in ["var", "all"]:
                try:
                    var_result = CausalityAnalyzer.var_analysis(
                        df_processed, lags, alpha, weight_metric
                    )
                    analysis_results["var"] = var_result
                except Exception as e:
                    error_msg = f"VAR analysis failed: {str(e)}"
                    logger.error(error_msg)
                    errors["var"] = error_msg
                    if method == "var":
                        raise HTTPException(status_code=500, detail=error_msg)
            
            if method in ["granger", "all"]:
                try:
                    granger_result = CausalityAnalyzer.granger_causality(
                        df_processed, max_lag, alpha
                    )
                    analysis_results["granger"] = granger_result
                except Exception as e:
                    error_msg = f"Granger causality analysis failed: {str(e)}"
                    logger.error(error_msg)
                    errors["granger"] = error_msg
                    if method == "granger":
                        raise HTTPException(status_code=500, detail=error_msg)
            
            if method in ["correlation", "all"]:
                try:
                    correlation_result = CausalityAnalyzer.correlation_analysis(
                        df_processed, correlation_threshold
                    )
                    analysis_results["correlation"] = correlation_result
                except Exception as e:
                    error_msg = f"Correlation analysis failed: {str(e)}"
                    logger.error(error_msg)
                    errors["correlation"] = error_msg
                    if method == "correlation":
                        raise HTTPException(status_code=500, detail=error_msg)
            
            if not analysis_results:
                raise HTTPException(
                    status_code=500,
                    detail=f"All analysis methods failed. Errors: {errors}"
                )
            
            response = {
                "validation": validation_result,
                "preprocessing": preprocessing_info,
                "analysis": analysis_results
            }
            
            if errors:
                response["partial_errors"] = errors
            
            return ensure_json_serializable(response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "causana",
        "version": "2.1.0"
    }

@app.get("/methods")
async def get_available_methods():
    """Get available analysis methods and parameters."""
    return {
        "methods": {
            "var": "Vector Autoregression - Tests Granger causality using VAR models",
            "granger": "Granger Causality Test - Direct pairwise Granger causality tests",
            "correlation": "Correlation Analysis - Finds highly correlated variable pairs",
            "all": "Run all available methods"
        },
        "weight_metrics": {
            "neglogp": "Negative log p-value (recommended)",
            "fstat": "F-statistic",
            "confidence": "1 - p-value"
        },
        "parameters": {
            "lags": "Number of lags for VAR (1-20)",
            "max_lag": "Maximum lag for Granger test (1-20)",
            "alpha": "Significance level (0-1)",
            "correlation_threshold": "Minimum correlation (0-1)",
            "normalize": "Standardize data (recommended: true)",
            "check_stationarity": "Test for stationarity (optional)"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Causana API",
        "version": "2.1.0",
        "description": "Causality analysis for time series data",
        "endpoints": {
            "/analyze": "POST - Upload CSV and perform causality analysis",
            "/health": "GET - Health check",
            "/methods": "GET - Available methods and parameters",
            "/docs": "GET - Interactive API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )