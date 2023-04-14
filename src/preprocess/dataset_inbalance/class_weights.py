import numpy as np


def label_to_index(metadata_df):
    unique_labels = sorted(metadata_df['primary_label'].unique())
    label_to_index_map = {label: index for index, label in enumerate(unique_labels)}
    return label_to_index_map


def calculate_class_weights(metadata_df, num_classes):
    class_counts = metadata_df['primary_label'].value_counts().to_dict()
    total_samples = sum(class_counts.values())
    class_weights = np.zeros(num_classes)

    for label, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        index = label_to_index[label]
        class_weights[index] = weight

    return class_weights

# How to use the class weights
# import torch.optim as optim
# Calculate class weights
# class_weights = calculate_class_weights(balanced_metadata_df, num_classes=264)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
# Set up the loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
# loss = criterion(output, torch.argmax(target, dim=1))
