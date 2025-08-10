# Causana: Advanced Causal Analysis & Visualization Platform

**Causana** is a professional, full-stack system for analyzing **multivariate time series data** to uncover **temporal causal relationships**. Users can upload datasets, run advanced causal inference algorithms, and visualize the results as **interactive Directed Acyclic Graphs (DAGs)** showing causality across variables and lags.

## 🌟 Features

### 🔬 Advanced Causal Analysis
- **VAR Analysis** - Vector Autoregression based causality detection
- **Granger Causality** - Statistical causality testing
- **Correlation Analysis** - Relationship strength measurement
- **Multiple Analysis Methods** - Run all methods simultaneously

### 📊 Professional Visualization
- **Interactive Network Graphs** - Drag, zoom, and explore causal relationships
- **Real-time Statistics** - Node count, edge density, average weights
- **Professional UI** - Modern, responsive design with dark/light mode
- **Graph Controls** - Fit to view, stabilize, node selection

### 🛡️ Robust Data Processing
- **Smart CSV Parsing** - Automatic delimiter detection
- **Date Column Handling** - Automatic conversion to numeric values
- **Data Validation** - Comprehensive quality checks and warnings
- **Missing Value Handling** - Intelligent imputation strategies
- **Linear Dependency Detection** - Automatic correlation-based column removal

### 🚀 Modern Architecture
- **FastAPI Backend** - High-performance Python API
- **React Frontend** - Modern, responsive web interface
- **Professional Styling** - CSS custom properties and animations
- **PWA Ready** - Progressive Web App capabilities

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend│    │   Data Processing│
│   (Port 1234)   │◄──►│   (Port 8000)   │◄──►│   & Analysis    │
│                 │    │                 │    │                 │
│ • File Upload   │    │ • CSV Processing│    │ • VAR Analysis  │
│ • Interactive   │    │ • Data Validation│    │ • Granger Tests │
│   Graphs        │    │ • Causal Analysis│    │ • Correlation   │
│ • Professional  │    │ • JSON Response │    │ • Matrix Checks │
│   UI            │    │ • Error Handling│    │ • Noise Addition│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Frontend
- **React 18** - Modern UI framework
- **react-graph-vis** - Interactive network visualization
- **Parcel** - Fast bundler
- **CSS Custom Properties** - Professional styling system

### Backend
- **FastAPI** - High-performance Python web framework
- **Pandas** - Data manipulation and analysis
- **Statsmodels** - Statistical analysis (VAR, Granger causality)
- **Scikit-learn** - Data preprocessing and normalization
- **NumPy** - Numerical computing

## 📦 Installation

### Prerequisites
- **Node.js** 16+ (for frontend)
- **Python** 3.9+ (for backend)
- **Git** (for version control)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/R0h-a-a-an/Causana.git
   cd Causana
   ```

2. **Setup Backend**
   ```bash
   cd causal-engine
   pip install -r requirements.txt
   python main.py
   ```

3. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Access the Application**
   - Frontend: http://localhost:1234
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🎯 Usage

### 1. Upload Your Data
- Drag and drop or click to browse for CSV files
- Supports various delimiters (comma, semicolon, tab, pipe)
- Automatic date column detection and conversion

### 2. Configure Analysis
- **Analysis Method**: VAR, Granger, Correlation, or All
- **Lags**: Number of time lags for VAR analysis (1-20)
- **Alpha**: Significance level for hypothesis testing (0.01-0.99)
- **Weight Metric**: Negative log p-value, F-statistic, or Confidence
- **Normalize**: Standardize data for better analysis

### 3. View Results
- **Validation Panel**: Data quality metrics and warnings
- **Interactive Graph**: Explore causal relationships
- **Statistics**: Node count, edge density, average weights
- **Graph Controls**: Fit to view, stabilize, select nodes

## 📊 Supported Data Formats

### CSV Requirements
- **Minimum**: 2 columns, 50 rows (recommended)
- **Maximum**: No strict limit (handles large datasets)
- **Data Types**: Numeric, date, text (auto-converted)
- **Missing Values**: Automatically handled
- **Delimiters**: Comma, semicolon, tab, pipe (auto-detected)

### Example Data Structure
```csv
datetime,nat_demand,T2M_toc,QV2M_toc,W2M_toc,Holiday_ID,school
2020-01-01 00:00,0,25.5,0.018,15.2,1,0
2020-01-01 01:00,0,24.8,0.017,14.9,1,0
...
```

## 🔧 API Endpoints

### POST `/analyze`
Main analysis endpoint for causal inference.

**Parameters:**
- `file`: CSV file upload
- `method`: Analysis method (`var`, `granger`, `correlation`, `all`)
- `lags`: Maximum lags for VAR (default: 5)
- `alpha`: Significance level (default: 0.05)
- `weight_metric`: Weight calculation (`neglogp`, `fstat`, `confidence`)
- `normalize`: Data normalization (default: true)
- `max_lag`: Maximum lag for Granger (default: 5)
- `correlation_threshold`: Correlation threshold (default: 0.7)

**Response:**
```json
{
  "validation": {
    "is_valid": true,
    "warnings": ["..."],
    "stats": {
      "rows": 744,
      "columns": 16,
      "numeric_columns": 14,
      "date_columns": 1,
      "text_columns": 1
    }
  },
  "analysis": {
    "var": {
      "nodes": ["X1", "X2", "X3"],
      "edges": [
        {
          "source": "X1",
          "target": "X2",
          "weight": 2.5,
          "p_value": 0.001,
          "lag": 2
        }
      ]
    }
  }
}
```

### GET `/health`
Health check endpoint.

### GET `/methods`
Available analysis methods and parameters.

## 🎨 Customization

### Styling
The frontend uses CSS custom properties for easy theming:
```css
:root {
  --primary-color: #6366f1;
  --bg-primary: #0f172a;
  --text-primary: #f8fafc;
  /* ... more variables */
}
```

### Analysis Parameters
Modify analysis behavior in `causal-engine/main.py`:
- Correlation thresholds
- Default lag values
- Significance levels
- Data preprocessing steps

## 🚀 Deployment

### Production Setup
1. **Build Frontend**
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy Backend**
   ```bash
   cd causal-engine
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Environment Variables**
   - Set production database connections
   - Configure CORS origins
   - Set logging levels

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Statsmodels** for statistical analysis capabilities
- **React Graph Vis** for network visualization
- **FastAPI** for high-performance API framework
- **Pandas** for data manipulation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/R0h-a-a-an/Causana/issues)
- **Discussions**: [GitHub Discussions](https://github.com/R0h-a-a-an/Causana/discussions)
- **Documentation**: [API Docs](http://localhost:8000/docs) (when running locally)

---

**Causana** - Unlock the hidden causal relationships in your time series data! 🔗📊
