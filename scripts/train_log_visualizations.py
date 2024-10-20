import pandas as pd
import matplotlib.pyplot as plt
import os

#Input run number
log_version = '01'

def plot_loss(train_log: pd.DataFrame, log_version:str):   
    """
    Plots the loss curve for a certain log of runtime information. 

    train_log: Dataframe containing the loss values
    log_version: String containing a log identifier
    """ 
    # Average all results and remove event nr
    train_log = train_log.groupby('epoch').mean()

    loss_avg_epoch = train_log['loss']
    plt.figure(figsize=(10,6))
    plt.plot(loss_avg_epoch, marker='o', linestyle='-', color='b')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)

    save_path = os.path.join('./figures', f'loss_curve_{log_version}')
    plt.savefig(save_path, dpi=300)


def plot_pred_curve(train_log: pd.DataFrame, log_version: str):
    """
    Plots the min-max range of predictions as well as the mean for the predictions and the ground truths.  

    train_log: Dataframe containing the loss values
    log_version: String containing a log identifier
    """ 
    
    train_log_avg = train_log.groupby('epoch').mean()

    # Create the plot
    plt.figure(figsize=(10,6))

    # Plot the shaded areas for pred_size_min/pred_size_max and gt_size_min/gt_size_max
    plt.fill_between(train_log_avg.index, train_log_avg['pred_size_min'], train_log_avg['pred_size_max'], color='lightblue', alpha=0.5, label='Prediction Range (min-max)')

    # # Plot the average lines for A and B
    plt.plot(train_log_avg.index, (train_log_avg['pred_size_min'] + train_log_avg['pred_size_max']) / 2, linestyle='-', color='blue', label='Prediction average')
    plt.plot(train_log_avg.index, (train_log_avg['gt_size_min'] + train_log_avg['gt_size_max']) / 2, linestyle='-', color='red', label='Ground truth Average')

    # Add labels and title
    plt.title('')
    plt.xlabel('Epoch')
    plt.ylabel('Nr of particles')
    plt.grid(True)

    # Add legend
    plt.legend()

    # Save the plot
    save_path = os.path.join('./figures', f'pred_range_curve{log_version}')
    plt.savefig(save_path, dpi=300)



def plot_MSE(train_log: pd.DataFrame, log_version:str):  
    """
    Plots the MSE for each of the epochs for the min and the max values. 

    train_log: Dataframe containing the loss values
    log_version: String containing a log identifier
    """ 
    
    train_log['MSE_min'] = ((train_log['pred_size_min'] - train_log['gt_size_min']) ** 2)
    train_log['MSE_max'] = ((train_log['pred_size_max'] - train_log['gt_size_max']) ** 2)

    train_log_avg = train_log.groupby('epoch').mean()

    # Plotting 'column1' and 'column2' against 'epoch'
    plt.figure(figsize=(10, 6))
    plt.plot(train_log_avg['MSE_min'], label='MSE Min', color='blue', linestyle='-', marker='o')
    plt.plot(train_log_avg['MSE_max'], label='MSE_ Max', color='red', linestyle='--', marker='s')

    # Adding title and labels
    plt.title('Min MSE and Max MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Values')

    # Adding a legend to differentiate between the lines
    plt.legend()

    # Display the grid
    plt.grid(True)

    # Save the plot
    save_path = os.path.join('./figures', f'MSE_curve_{log_version}')
    plt.savefig(save_path, dpi=300)

if __name__ == '__main__':
    
    # Somehow introduce a unique identifier here
    train_log = pd.read_csv(f'./results/log_{log_version}.csv')
    train_log.drop(columns=['event'], inplace=True)

    plot_loss(train_log=train_log, log_version=log_version)
    plot_pred_curve(train_log=train_log, log_version=log_version)
    plot_MSE(train_log=train_log, log_version=log_version)

