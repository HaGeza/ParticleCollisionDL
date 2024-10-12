import pandas as pd
import matplotlib.pyplot as plt

train_log = pd.read_csv('./results/Analysis Dev/log.csv')

# Average all results and remove event nr
train_log.drop(columns=['event'], inplace=True)
averaged_df = train_log.groupby('epoch').mean()

# Plot loss curves
# loss_avg_epoch = averaged_df['loss']
# plt.figure(figsize=(10,6))
# plt.plot(loss_avg_epoch, marker='o', linestyle='-', color='b')
# plt.title('Average Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# plt.grid(True)
# plt.show()

# Plot the averaged min and max for prediction and ground truth

# Create the plot
plt.figure(figsize=(10,6))

# Plot the shaded areas for pred_size_min/pred_size_max and gt_size_min/gt_size_max
plt.fill_between(averaged_df.index, averaged_df['pred_size_min'], averaged_df['pred_size_max'], color='lightblue', alpha=0.5, label='A Range (min-max)')
plt.fill_between(averaged_df.index, averaged_df['gt_size_min'], averaged_df['gt_size_max'], color='lightcoral', alpha=0.5, label='B Range (min-max)')

# Plot the average lines for A and B
plt.plot(averaged_df.index, (averaged_df['pred_size_min'] + averaged_df['pred_size_max']) / 2, linestyle='-', color='blue', label='A Average')
plt.plot(averaged_df.index, (averaged_df['gt_size_min'] + averaged_df['gt_size_max']) / 2, linestyle='-', color='red', label='B Average')

# Plot the average loss per epoch
plt.plot(averaged_df.index, averaged_df['loss'], marker='o', linestyle='-', color='black', label='Average Loss')

# Add labels and title
plt.title('Average Loss and Ranges for A and B over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Values')
plt.grid(True)

# Add legend
plt.legend()

# Show the plot
plt.show()