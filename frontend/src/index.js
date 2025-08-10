import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import Graph from 'react-graph-vis';

// Professional loading spinner component
const LoadingSpinner = ({ message = "Processing..." }) => (
    <div className="loading-overlay">
        <div className="loading-spinner">
            <div className="spinner"></div>
            <p>{message}</p>
        </div>
    </div>
);

// Professional file upload component with drag & drop
const FileUpload = ({ onFileSelect, file }) => {
    const [isDragOver, setIsDragOver] = useState(false);
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragOver(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragOver(false);
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            onFileSelect(files[0]);
        }
    };

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (file) {
            onFileSelect(file);
        }
    };

    return (
        <div 
            className={`file-upload ${isDragOver ? 'drag-over' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
        >
            <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
            />
            <div className="upload-content">
                <div className="upload-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7,10 12,15 17,10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                </div>
                {file ? (
                    <div className="file-info">
                        <p className="file-name">{file.name}</p>
                        <p className="file-size">{(file.size / 1024).toFixed(1)} KB</p>
                    </div>
                ) : (
                    <div className="upload-text">
                        <p className="upload-title">Upload your CSV file</p>
                        <p className="upload-subtitle">Drag and drop or click to browse</p>
                    </div>
                )}
            </div>
        </div>
    );
};

// Enhanced parameter input component
const ParameterInput = ({ label, type = "text", options, value, onChange, min, max, step, placeholder }) => {
    if (type === "select") {
        return (
            <div className="parameter-group">
                <label>{label}</label>
                <select value={value} onChange={(e) => onChange(e.target.value)}>
                    {options.map(option => (
                        <option key={option.value} value={option.value}>
                            {option.label}
                        </option>
                    ))}
                </select>
            </div>
        );
    }

    return (
        <div className="parameter-group">
            <label>{label}</label>
            <input
                type={type}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                min={min}
                max={max}
                step={step}
                placeholder={placeholder}
            />
        </div>
    );
};

// Professional graph viewer with enhanced features
const GraphViewer = ({ graphData, analysisMethod, onMethodChange }) => {
    const [network, setNetwork] = useState(null);
    const [selectedNode, setSelectedNode] = useState(null);
    const [graphStats, setGraphStats] = useState({});

    useEffect(() => {
        if (network) {
            network.setOptions({ physics: true });
        }
    }, [graphData, network]);

    useEffect(() => {
        if (graphData && graphData.nodes && graphData.edges) {
            const stats = {
                nodes: graphData.nodes.length,
                edges: graphData.edges.length,
                density: graphData.edges.length / (graphData.nodes.length * (graphData.nodes.length - 1)),
                avgWeight: graphData.edges.reduce((sum, edge) => sum + Math.abs(edge.weight), 0) / graphData.edges.length
            };
            setGraphStats(stats);
        }
    }, [graphData]);

    if (!graphData || !graphData.nodes || !graphData.edges) {
        return (
            <div className="no-data">
                <div className="no-data-icon">📊</div>
                <h3>No Analysis Data</h3>
                <p>Upload a CSV file and run analysis to see the causal graph</p>
            </div>
        );
    }

    const { nodes, edges } = graphData;

    const transformedNodes = nodes.map(node => ({
        id: node,
        label: node,
        title: node
    }));

    const transformedLinks = edges.map(link => {
        let label = `W: ${link.weight.toFixed(2)}`;
        if (link.lag) {
            label += `, L: ${link.lag}`;
        }
        if (link.p_value !== undefined) {
            label += `, p: ${link.p_value.toExponential(2)}`;
        }
        if (link.correlation !== undefined) {
            label += `, r: ${link.correlation.toFixed(3)}`;
        }
        return {
            from: link.source,
            to: link.target,
            label: label,
            title: label
        };
    });

    const graph = {
        nodes: transformedNodes,
        edges: transformedLinks
    };

    const options = {
        layout: {
            hierarchical: false,
            improvedLayout: true
        },
        edges: {
            color: '#64B5F6',
            width: 2,
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 0.8
                }
            },
            font: {
                color: '#ffffff',
                size: 11,
                face: 'Arial'
            },
            smooth: {
                type: 'continuous',
                forceDirection: 'none'
            }
        },
        nodes: {
            shape: 'circle',
            size: 30,
            font: {
                color: '#ffffff',
                size: 14,
                face: 'Arial',
                bold: 'bold'
            },
            color: {
                background: '#4CAF50',
                border: '#2E7D32',
                highlight: {
                    background: '#66BB6A',
                    border: '#388E3C'
                }
            },
            borderWidth: 2
        },
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -100,
                centralGravity: 0.01,
                springConstant: 0.08,
                springLength: 200,
                damping: 0.4,
                avoidOverlap: 1
            },
            minVelocity: 0.75,
            stabilization: {
                enabled: true,
                iterations: 1000,
                updateInterval: 100
            }
        },
        interaction: {
            dragNodes: true,
            dragView: true,
            zoomView: true,
            hover: true,
            tooltipDelay: 200
        }
    };

    const events = {
        stabilized: () => {
            if (network) {
                network.setOptions({ physics: false });
            }
        },
        click: (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                setSelectedNode(nodeId);
            } else {
                setSelectedNode(null);
            }
        }
    };

    return (
        <div className="graph-viewer">
            <div className="graph-header">
                <div className="graph-title">
                    <h3>{analysisMethod} Analysis</h3>
                    <div className="graph-stats">
                        <span className="stat">
                            <span className="stat-label">Nodes:</span>
                            <span className="stat-value">{graphStats.nodes}</span>
                        </span>
                        <span className="stat">
                            <span className="stat-label">Edges:</span>
                            <span className="stat-value">{graphStats.edges}</span>
                        </span>
                        <span className="stat">
                            <span className="stat-label">Density:</span>
                            <span className="stat-value">{(graphStats.density * 100).toFixed(1)}%</span>
                        </span>
                        <span className="stat">
                            <span className="stat-label">Avg Weight:</span>
                            <span className="stat-value">{graphStats.avgWeight?.toFixed(2) || 'N/A'}</span>
                        </span>
                    </div>
                </div>
                <div className="graph-controls">
                    <button 
                        className="control-btn"
                        onClick={() => network?.fit()}
                        title="Fit to view"
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                            <circle cx="8.5" cy="8.5" r="1.5"/>
                            <polyline points="21,15 16,10 5,21"/>
                        </svg>
                    </button>
                    <button 
                        className="control-btn"
                        onClick={() => network?.stabilize()}
                        title="Stabilize"
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="3"/>
                            <path d="M12 1v6m0 6v6"/>
                            <path d="M1 12h6m6 0h6"/>
                        </svg>
                    </button>
                </div>
            </div>
            
            {selectedNode && (
                <div className="node-info">
                    <h4>Selected Node: {selectedNode}</h4>
                    <p>Click on the graph to select different nodes</p>
                </div>
            )}
            
            <div className="graph-container">
                <Graph
                    graph={graph}
                    options={options}
                    events={events}
                    getNetwork={setNetwork}
                    style={{ height: "600px", backgroundColor: "#1a1a1a" }}
                />
            </div>
        </div>
    );
};

// Enhanced validation info component
const ValidationInfo = ({ validation }) => {
    if (!validation) return null;

    const getStatusColor = (isValid) => isValid ? '#4CAF50' : '#f44336';
    const getStatusIcon = (isValid) => isValid ? '✅' : '❌';

    return (
        <div className="validation-panel">
            <div className="validation-header">
                <h3>Data Validation</h3>
                <div className="validation-status">
                    <span className="status-icon">{getStatusIcon(validation.is_valid)}</span>
                    <span className="status-text" style={{ color: getStatusColor(validation.is_valid) }}>
                        {validation.is_valid ? 'Valid' : 'Invalid'}
                    </span>
                </div>
            </div>
            
            <div className="validation-grid">
                <div className="validation-card">
                    <h4>📊 Data Structure</h4>
                    <div className="stat-row">
                        <span>Rows:</span>
                        <span className="stat-value">{validation.stats?.rows}</span>
                    </div>
                    <div className="stat-row">
                        <span>Total Columns:</span>
                        <span className="stat-value">{validation.stats?.columns}</span>
                    </div>
                    <div className="stat-row">
                        <span>Processed Columns:</span>
                        <span className="stat-value">{validation.stats?.expected_numeric_after_preprocessing}</span>
                    </div>
                </div>
                
                <div className="validation-card">
                    <h4>🔢 Column Types</h4>
                    <div className="stat-row">
                        <span>Numeric:</span>
                        <span className="stat-value">{validation.stats?.numeric_columns}</span>
                    </div>
                    <div className="stat-row">
                        <span>Date:</span>
                        <span className="stat-value">{validation.stats?.date_columns}</span>
                    </div>
                    <div className="stat-row">
                        <span>Text:</span>
                        <span className="stat-value">{validation.stats?.text_columns}</span>
                    </div>
                </div>
                
                <div className="validation-card">
                    <h4>⚠️ Data Quality</h4>
                    <div className="stat-row">
                        <span>Missing Values:</span>
                        <span className="stat-value">{validation.stats?.missing_values}</span>
                    </div>
                    <div className="stat-row">
                        <span>Infinite Values:</span>
                        <span className="stat-value">{validation.stats?.infinite_values}</span>
                    </div>
                </div>
            </div>
            
            {validation.warnings && validation.warnings.length > 0 && (
                <div className="warnings-panel">
                    <h4>⚠️ Warnings</h4>
                    <div className="warnings-list">
                        {validation.warnings.map((warning, index) => (
                            <div key={index} className="warning-item">
                                <span className="warning-icon">⚠️</span>
                                <span>{warning}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

// Main analysis function
const analyzeCausality = async (file, method, lags, alpha, weightMetric, normalize, maxLag, correlationThreshold) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("method", method);
    formData.append("lags", String(lags));
    formData.append("alpha", String(alpha));
    formData.append("weight_metric", weightMetric);
    formData.append("normalize", String(normalize));
    formData.append("max_lag", String(maxLag));
    formData.append("correlation_threshold", String(correlationThreshold));

    const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorText}`);
    }

    return response.json();
};

// Main App component
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
            console.log("Analysis results:", result);
        } catch (error) {
            console.error("Analysis failed", error);
            let errorMessage = error.message;
            if (error.message.includes("Server error:")) {
                try {
                    const errorText = error.message.split("Server error:")[1];
                    const errorData = JSON.parse(errorText);
                    errorMessage = errorData.detail || errorText;
                } catch (e) {
                    errorMessage = error.message;
                }
            }
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
            <header className="app-header">
                <div className="header-content">
                    <div className="logo">
                        <div className="logo-icon">🔗</div>
                        <h1>Causana</h1>
                    </div>
                    <p className="tagline">Advanced Causal Analysis & Visualization</p>
                </div>
            </header>

            <main className="app-main">
                <div className="content-wrapper">
                    <section className="upload-section">
                        <div className="section-header">
                            <h2>Data Upload & Analysis</h2>
                            <p>Upload your CSV file and configure analysis parameters</p>
                        </div>
                        
                        <form onSubmit={handleSubmit} className="analysis-form">
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
                            
                            <button type="submit" className="analyze-btn" disabled={loading || !selectedFile}>
                                {loading ? (
                                    <>
                                        <div className="btn-spinner"></div>
                                        Analyzing...
                                    </>
                                ) : (
                                    <>
                                        <span className="btn-icon">🔍</span>
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
                                <div key={method} className="analysis-result">
                                    <GraphViewer 
                                        graphData={data} 
                                        analysisMethod={method.toUpperCase()} 
                                    />
                                </div>
                            ))}
                        </section>
                    )}
                </div>
            </main>

            {loading && <LoadingSpinner message="Processing your data..." />}
        </div>
    );
}

const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />); 