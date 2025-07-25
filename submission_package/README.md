# PM10 Time Series Analysis - Competition Submission

## 🏆 Team Solution: Advanced Fourier Decomposition with Statistical Anomaly Detection

### 📋 Overview

This repository contains our comprehensive solution for the PM10 air quality time series analysis competition. Our approach combines advanced signal processing techniques with statistical analysis to provide deep insights into air pollution patterns in Kraków, Poland.

### 🎯 Key Features

- **Advanced Time Series Analysis**: Fourier decomposition for trend and seasonal pattern extraction
- **Statistical Anomaly Detection**: Z-score based outlier identification
- **Object-Oriented Architecture**: Clean, maintainable, and extensible code structure
- **High Performance**: Optimized for <5 minute execution on single-core CPU with 2GB RAM
- **Comprehensive Output**: CSV files, visualizations, and detailed metrics

### 🏗️ Architecture

Our solution follows a robust object-oriented design pattern with the following key components:

```
PM10AnalysisSystem (Main Orchestrator)
├── DataLoader              # Data loading and preprocessing
├── TimeSeriesAnalyzer      # Fourier decomposition and trend analysis
├── StationAnalyzer         # Individual station analysis
├── OverallAnalyzer         # City-wide pattern analysis
├── OutputManager           # CSV and metrics generation
├── Visualizer              # Chart and graph generation
└── ReportGenerator         # Comprehensive reporting
```

### 📊 Technical Approach

#### 1. **Data Processing**
- **Format**: XLSX files only (CSV support removed for simplicity)
- **Memory Optimization**: Limited row processing (10,000 rows max per file)
- **Data Cleaning**: Automatic handling of missing values and type conversions
- **Station Filtering**: Focus on common stations across all years

#### 2. **Time Series Analysis**
- **Trend Extraction**: Linear regression on time series data
- **Seasonal Decomposition**: Fast Fourier Transform (FFT) for pattern identification
- **Frequency Analysis**: Multi-component seasonal signal reconstruction
- **Residual Analysis**: Noise and irregular pattern identification

#### 3. **Anomaly Detection**
- **Method**: Statistical z-score analysis
- **Threshold**: Configurable (default: 2.5 standard deviations)
- **Robustness**: Handles seasonal variations and trend effects

#### 4. **Pattern Recognition**
- **Monthly Patterns**: Detailed month-by-month analysis
- **Seasonal Trends**: Quarterly pattern identification
- **Long-term Trends**: Multi-year trend coefficient calculation

### 🚀 Quick Start

#### Prerequisites
```bash
Python 3.7+
pandas >= 1.3.0
numpy >= 1.20.0
matplotlib >= 3.3.0
scipy >= 1.7.0
openpyxl >= 3.0.0
```

#### Installation
```bash
# Clone or download the submission package
cd submission_package

# Install dependencies
pip install -r requirements.txt
```

#### Execution
```bash
# Run the complete analysis
python run_model.py
```

### 📁 Project Structure

```
submission_package/
├── run_model.py              # Main execution script
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── output/                   # Generated CSV files and metrics
│   ├── station_analysis.csv  # Individual station results
│   ├── monthly_patterns.csv  # Monthly pollution patterns
│   └── evaluation_metrics.json # Competition metrics
└── visualizations/           # Generated plots and charts
    ├── overall_trends.png    # City-wide trend analysis
    ├── station_comparison.png # Station-by-station comparison
    └── seasonal_patterns.png  # Seasonal and quarterly patterns
```

### 📈 Output Files

#### CSV Files (output/)
1. **station_analysis.csv** - Comprehensive analysis for each monitoring station
   - Basic statistics (mean, std, min, max, median)
   - Trend coefficients and seasonal amplitudes
   - Anomaly counts and data coverage metrics
   - Seasonal statistics for each quarter

2. **monthly_patterns.csv** - Monthly pollution patterns across all stations
   - Monthly averages, standard deviations, and ranges
   - Data availability and quality metrics

3. **evaluation_metrics.json** - Competition evaluation metrics
   - Execution time and performance metrics
   - Analysis quality indicators
   - Data coverage and processing statistics

#### Visualizations (visualizations/)
1. **overall_trends.png** - City-wide pollution trends over time
2. **station_comparison.png** - Comparative analysis between stations
3. **seasonal_patterns.png** - Seasonal and quarterly pattern analysis

### 🔬 Algorithm Details

#### Fourier Decomposition Process
1. **Preprocessing**: Data cleaning and normalization
2. **Detrending**: Linear trend removal using signal.detrend()
3. **FFT Analysis**: Fast Fourier Transform for frequency domain analysis
4. **Component Selection**: Top N frequency components (configurable)
5. **Reconstruction**: Inverse FFT for seasonal component recovery
6. **Residual Calculation**: Original - Trend - Seasonal

#### Anomaly Detection Algorithm
1. **Z-Score Calculation**: (value - mean) / standard_deviation
2. **Threshold Application**: |z-score| > threshold (default: 2.5)
3. **Temporal Context**: Considers seasonal variations
4. **Robustness**: Handles non-normal distributions

### ⚡ Performance Optimizations

- **Memory Management**: Row-limited data loading (10K rows max)
- **Computational Efficiency**: Limited station analysis (7 stations max)
- **Vectorized Operations**: NumPy and pandas optimizations
- **Smart Sampling**: Intelligent data subsampling for large datasets
- **Single-Core Design**: No parallel processing overhead

### 📊 Key Insights Delivered

1. **Trend Analysis**
   - Overall city pollution trend (increasing/decreasing)
   - Individual station trend coefficients
   - Long-term pattern identification

2. **Seasonal Patterns**
   - Monthly pollution profiles
   - Quarterly seasonal variations
   - Winter vs. summer comparisons

3. **Anomaly Detection**
   - Unusual pollution events identification
   - Station-specific anomaly patterns
   - Temporal anomaly distribution

4. **Comparative Analysis**
   - Station-by-station performance
   - Spatial pollution variations
   - Data quality assessments

### 🎯 Competition Compliance

- ✅ **Execution Time**: <5 minutes on single-core CPU
- ✅ **Memory Usage**: <2GB RAM optimized
- ✅ **Code Quality**: Object-oriented, well-documented
- ✅ **Output Format**: CSV files with proper metrics
- ✅ **Visualization**: Professional charts and graphs
- ✅ **Reproducibility**: Complete dependency management

### 🔧 Configuration

Key parameters can be adjusted in the `ProjectConfig` class:

```python
# Data processing parameters
self.max_rows_per_file = 10000    # Memory optimization
self.max_stations = 7             # Computational efficiency

# Analysis parameters
self.fourier_components = 3       # FFT decomposition depth
self.anomaly_threshold = 2.5      # Z-score threshold
self.min_data_points = 30         # Minimum data for analysis

# Performance constraints
self.max_execution_time = 300     # 5 minutes in seconds
```

### 🏆 Expected Results

The analysis typically identifies:
- **Seasonal Patterns**: Clear winter/summer pollution cycles
- **Trend Direction**: Long-term improvement or deterioration
- **Station Variability**: Different pollution levels across monitoring points
- **Anomalous Events**: Unusual pollution spikes or drops
- **Monthly Profiles**: Detailed month-by-month characteristics

### 📞 Support

For questions about the implementation or results:
- Review the generated `evaluation_metrics.json` for performance details
- Check the `output/` directory for detailed CSV results
- Examine the `visualizations/` directory for graphical insights

### 🏅 Competitive Advantages

1. **Robust Algorithm**: Handles missing data and irregular patterns
2. **Scalable Design**: Easy to extend for additional analysis types
3. **Performance Optimized**: Meets strict time and memory constraints
4. **Professional Output**: Competition-ready results and visualizations
5. **Well Documented**: Clear code structure and comprehensive documentation

---

**Team**: Data Mining Experts  
**Competition**: PM10 Time Series Analysis Hackathon  
**Date**: July 2025  
**Version**: 1.0
