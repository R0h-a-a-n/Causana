export const analyzeCausality = async (
    file, 
    method, 
    lags, 
    alpha, 
    weightMetric, 
    normalize, 
    maxLag, 
    correlationThreshold
) => {
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
        try {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        } catch (e) {
            throw new Error(`Server error: ${response.status}`);
        }
    }

    return response.json();
};