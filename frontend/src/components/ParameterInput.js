import React from 'react';

function ParameterInput({ label, type = "text", options, value, onChange, min, max, step }) {
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
            />
        </div>
    );
}

export default ParameterInput;