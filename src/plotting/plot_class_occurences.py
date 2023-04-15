import matplotlib.pyplot as plt


def plot_class_occurrences(metadata_df):
    class_counts = metadata_df['primary_label'].value_counts().to_dict()

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(class_counts.keys(), class_counts.values())
    ax.set_xlabel('Class')
    ax.set_ylabel('Occurrences')
    ax.set_title('Occurrences of Classes in the Balanced Dataset')
    plt.xticks(rotation=90)
    plt.show()
