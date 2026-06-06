# Pitcher Strikeout Prediction Model

A machine learning model that predicts daily pitcher strikeouts using comprehensive statistics, pitcher handedness, and batter matchup data.

## Features

- **Advanced Feature Engineering**: Uses pitcher stats (K/9, ERA, velocity), handedness matchups (L/R), and batter tendencies
- **Daily Predictions**: Generate strikeout predictions from morning lineups
- **Results Tracking**: Log actual results each night for continuous model improvement
- **Continuous Learning**: Automatically retrains model with new performance data
- **Model Performance Analytics**: Track accuracy, precision, and ROI over time

## Project Structure

```
project/
├── model/
│   ├── strikeout_predictor.py      # Main prediction model
│   ├── features.py                 # Feature engineering
│   └── training.py                 # Training pipeline
├── data/
│   ├── lineups/                    # Daily lineup inputs
│   ├── results/                    # Nightly results
│   └── model_history/              # Model performance tracking
├── utils/
│   ├── data_processor.py           # Data cleaning
│   └── stats_calculator.py         # Stat calculations
├── daily_predictions.py            # Generate daily predictions
├── results_tracker.py              # Log nightly results
├── model_updater.py                # Retrain with new data
├── config.py                       # Configuration
└── requirements.txt                # Dependencies
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure settings in `config.py`
4. Place daily lineups in `data/lineups/`
5. Run predictions and track results

## Usage

### Generate Daily Predictions
```bash
python daily_predictions.py --lineup data/lineups/today.csv
```

### Log Results
```bash
python results_tracker.py --results data/results/today_results.csv
```

### Retrain Model
```bash
python model_updater.py
```

## Model Performance

View analytics in `data/model_history/performance_log.csv`
