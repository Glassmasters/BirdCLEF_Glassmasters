import pandas as pd


from characteristics import audio_waveform_maximum, audio_waveform_minimum, audio_waveform_mean, audio_waveform_std, \
    load_audio_waveform_from_folder
from models.knn import KNN
from models.neuronalnetwork import NeuronalNetwork
from models.linearregression import LinearRegressionModel


def load_into_list(audio_datas):
    """

    Args:
        audio_datas: list of tuples, where each tuple contains the audio waveform and the label.

    Returns: audio_characteristics: list of lists, where each list contains the audio characteristics of the audio
    Returns: y: list of labels

    """
    audio_characteristics = []
    y = []
    for audio_data in audio_datas:
        audio_characteristics.append([audio_waveform_maximum(audio_data[0]),
                                      audio_waveform_minimum(audio_data[0]),
                                      audio_waveform_mean(audio_data[0]),
                                      audio_waveform_std(audio_data[0])])
        y.append(audio_data[1])
        del audio_data
    return audio_characteristics, y


def train_fit_predict(modeltype: str):
    # Load audio waveform data
    print("Loading audio waveform data...")
    path = "../../birdclef-2023/train_metadata.csv"
    audio_datas = load_audio_waveform_from_folder(path)
    test_data = load_audio_waveform_from_folder(
        '/home/meri/Documents/GitHub/BirdCLEF_Glassmasters/data/test_metadata.csv')
    species = \
        pd.read_csv("/home/meri/Documents/GitHub/BirdCLEF_Glassmasters/src/birdclef-2023/eBird_Taxonomy_v2021.csv")[
            "SPECIES_CODE"].tolist()

    # Create dictionary to map species to indicator values
    species_dict = {species[i]: i for i in range(len(species))}
    # Extract audio characteristics
    print("Extracting audio characteristics...")
    audio_characteristics, y = load_into_list(audio_datas)

    # Replace species in target variable with indicator values
    y = [species_dict[val] for val in y]
    print(y)
    # Fit model and evaluate performance
    print("Fitting model and evaluating performance...")
    match modeltype:
        case "linear":
            model = LinearRegressionModel()
        case "knn":
            model = KNN()
        case "neuronalnetwork":
            model = NeuronalNetwork()
        case _: model = LinearRegressionModel()

    model.fit(audio_characteristics, y)
    print("Extracting test audio characteristics...")
    test_characteristics, y_test = load_into_list(test_data)
    # Replace species in target variable with indicator values
    y_test = [species_dict[val] for val in y_test]
    y_pred = list(map(int, model.predict(test_characteristics)))
    print(y_pred)
    print(y_test)
    # Count number of correct predictions
    num_correct = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            num_correct += 1
    # Print results
    print(f"Model score after fitting: {model.score(test_characteristics, y_test)}")
    print(f"Number of correct predictions: {num_correct}/{len(y_test)}")


if __name__ == '__main__':
    train_fit_predict("knn")
