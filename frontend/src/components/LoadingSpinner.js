import React from 'react';

function LoadingSpinner({ message = "Processing..." }) {
    return (
        <div className="loading-overlay">
            <div className="loading-content">
                <div className="spinner"></div>
                <p>{message}</p>
            </div>
        </div>
    );
}

export default LoadingSpinner;