This project uses an LSTM-based Variational Autoencoder (VAE) to detect "doubling points" (where a stock price doubles within a lookahead window) and anomaly signals in Nepali stock OHLCV (Open, High, Low, Close, Volume) data.

## Features

- **Doubling Point Detection:** Automatically finds points where a stock price doubles within a specified future window.
- **Deep Learning:** Implements an LSTM-VAE in PyTorch to learn normal price patterns and spot anomalies.
- **Multi-Stock Support:** Processes and analyzes multiple stocks in batch.
- **Signal Analysis:** Uses VAE reconstruction error to generate buy signals and evaluate their effectiveness.
- **Visualization:** Plots doubling points, buy signals, and error distributions for actionable insights.

## Workflow

1. **Data Preparation:**  
   - Reads OHLCV CSV files for each stock.
   - Detects doubling points and extracts windows of data before each event.
   - Scales features for model input.

2. **Model Training:**  
   - Trains an LSTM-VAE on sequences ending at doubling points.

3. **Evaluation & Visualization:**  
   - Computes reconstruction error for each sequence.
   - Analyzes error distribution per stock.
   - Generates buy signals based on low reconstruction error.
   - Evaluates future returns for these signals.
   - Visualizes results.

## Usage

1. **Prepare Data:**  
   Place your OHLCV CSV files (with columns: `date, open, high, low, close, volume`) in the `datas/` directory.

2. **Run the Notebook:**  
   Open `VAE_OHLCV.ipynb` and run all cells.  
   The notebook will:
   - Detect doubling points
   - Train the LSTM-VAE
   - Analyze and visualize results

3. **Outputs:**  
   - `doubled_stocks_info.csv`: Summary of all detected doubling events.
   - Plots for doubling points, reconstruction errors, and buy signals.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- torch
- matplotlib
- seaborn

Install dependencies with:
```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn
```

## Notes

- All data used is public and for research purposes only.
- The code is for educational and research use; not financial advice.
