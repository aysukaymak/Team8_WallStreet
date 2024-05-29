import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cat = pd.read_csv("outputs/cat_results.csv")
xgb = pd.read_csv("outputs/xgb_results.csv")
lgb = pd.read_csv("outputs/lgb_results.csv")
all = pd.read_csv("outputs/results.csv")

def plot_results(df, model_name):
    # Set the trial column as the index
    df.set_index('trial', inplace=True)

    # Plot the metrics over the trials
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['AUC'], label='AUC')
    plt.plot(df.index, df['F1'], label='F1')
    plt.plot(df.index, df['Recall'], label='Recall')
    plt.plot(df.index, df['Precision'], label='Precision')
    plt.plot(df.index, df['Accuracy'], label='Accuracy')

    # Add titles and labels
    plt.title(f'Performance Metrics over Trials for {model_name}')
    plt.xlabel('Trial')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/{model_name}_results.png', bbox_inches='tight')

#plot_results(cat, 'CatBoost')
#plot_results(xgb, 'XGBoost')
plot_results(lgb, 'LightGBM')
plot_results(all, 'Ensemble Model')
