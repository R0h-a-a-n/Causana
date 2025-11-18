import React, { useState } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import ParameterInput from './components/ParameterInput';
import ValidationInfo from './components/ValidationInfo';
import GraphViewer from './components/GraphViewer';
import LoadingSpinner from './components/LoadingSpinner';
import { analyzeCausality } from './api/causalityApi';

function App() {
    const [analysisResults, setAnalysisResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [parameters, setParameters] = useState({
        method: 'var',
        lags: '5',
        alpha: '0.05',
        weightMetric: 'neglogp',
        normalize: 'true',
        maxLag: '5',
        correlationThreshold: '0.7'
    });

    const handleFileSelect = (file) => {
        setSelectedFile(file);
        setError(null);
        setAnalysisResults(null);
    };

    const handleParameterChange = (key, value) => {
        setParameters(prev => ({ ...prev, [key]: value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!selectedFile) {
            setError("Please select a CSV file");
            return;
        }

        setLoading(true);
        setError(null);
        setAnalysisResults(null);

        try {
            const result = await analyzeCausality(
                selectedFile,
                parameters.method,
                Number(parameters.lags),
                Number(parameters.alpha),
                parameters.weightMetric,
                parameters.normalize === 'true',
                Number(parameters.maxLag),
                Number(parameters.correlationThreshold)
            );
            setAnalysisResults(result);
        } catch (error) {
            console.error("Analysis failed", error);
            setError(error.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
            <Header />

            <main className="main-content">
                <section className="upload-section">
                    <div className="section-header">
                        <h2>Data Upload & Analysis</h2>
                        <p>Upload your CSV file and configure analysis parameters</p>
                    </div>
                    
                    <form onSubmit={handleSubmit}>
                        <div className="form-section">
                            <h3>📁 Data File</h3>
                            <FileUpload onFileSelect={handleFileSelect} file={selectedFile} />
                        </div>
                        
                        <div className="form-section">
                            <h3>⚙️ Analysis Configuration</h3>
                            <div className="parameters-grid">
                                <ParameterInput
                                    label="Analysis Method"
                                    type="select"
                                    options={[
                                        { value: 'var', label: 'VAR Analysis' },
                                        { value: 'granger', label: 'Granger Causality' },
                                        { value: 'correlation', label: 'Correlation Analysis' },
                                        { value: 'all', label: 'All Methods' }
                                    ]}
                                    value={parameters.method}
                                    onChange={(value) => handleParameterChange('method', value)}
                                />
                                
                                <ParameterInput
                                    label="Weight Metric"
                                    type="select"
                                    options={[
                                        { value: 'neglogp', label: 'Negative Log P-value' },
                                        { value: 'fstat', label: 'F-statistic' },
                                        { value: 'confidence', label: 'Confidence' }
                                    ]}
                                    value={parameters.weightMetric}
                                    onChange={(value) => handleParameterChange('weightMetric', value)}
                                />
                                
                                <ParameterInput
                                    label="Lags"
                                    type="number"
                                    value={parameters.lags}
                                    onChange={(value) => handleParameterChange('lags', value)}
                                    min="1"
                                    max="20"
                                />
                                
                                <ParameterInput
                                    label="Alpha (Significance)"
                                    type="number"
                                    value={parameters.alpha}
                                    onChange={(value) => handleParameterChange('alpha', value)}
                                    min="0.01"
                                    max="0.99"
                                    step="0.01"
                                />
                                
                                <ParameterInput
                                    label="Max Lag (Granger)"
                                    type="number"
                                    value={parameters.maxLag}
                                    onChange={(value) => handleParameterChange('maxLag', value)}
                                    min="1"
                                    max="20"
                                />
                                
                                <ParameterInput
                                    label="Correlation Threshold"
                                    type="number"
                                    value={parameters.correlationThreshold}
                                    onChange={(value) => handleParameterChange('correlationThreshold', value)}
                                    min="0.1"
                                    max="1.0"
                                    step="0.1"
                                />
                                
                                <ParameterInput
                                    label="Normalize Data"
                                    type="select"
                                    options={[
                                        { value: 'true', label: 'Yes' },
                                        { value: 'false', label: 'No' }
                                    ]}
                                    value={parameters.normalize}
                                    onChange={(value) => handleParameterChange('normalize', value)}
                                />
                            </div>
                        </div>
                        
                        <button 
                            type="submit" 
                            className="analyze-btn"
                            disabled={loading || !selectedFile}
                        >
                            {loading ? (
                                <>
                                    <div className="btn-spinner"></div>
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <span>🔍</span>
                                    Run Analysis
                                </>
                            )}
                        </button>
                    </form>
                </section>

                {error && (
                    <section className="error-section">
                        <div className="error-card">
                            <div className="error-header">
                                <span className="error-icon">❌</span>
                                <h3>Analysis Error</h3>
                            </div>
                            <p>{error}</p>
                        </div>
                    </section>
                )}

                {analysisResults && (
                    <section className="results-section">
                        <div className="section-header">
                            <h2>Analysis Results</h2>
                            <p>Detailed insights from your causal analysis</p>
                        </div>
                        
                        <ValidationInfo validation={analysisResults.validation} />
                        
                        {analysisResults.analysis && Object.entries(analysisResults.analysis).map(([method, data]) => (
                            <GraphViewer 
                                key={method}
                                graphData={data} 
                                analysisMethod={method.toUpperCase()} 
                            />
                        ))}
                    </section>
                )}
            </main>

            {loading && <LoadingSpinner message="Processing your data... This may take a moment for large datasets." />}
        </div>
    );
}

export default App;