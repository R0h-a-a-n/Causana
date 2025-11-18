import React from 'react';

function ValidationInfo({ validation }) {
    if (!validation) return null;

    return (
        <div className="validation-panel">
            <div className="validation-header">
                <h3>Data Validation</h3>
                <div className="validation-status">
                    <span className="status-icon">{validation.is_valid ? '✅' : '❌'}</span>
                    <span 
                        className="status-text" 
                        style={{ color: validation.is_valid ? '#4CAF50' : '#f44336' }}
                    >
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
}

export default ValidationInfo;