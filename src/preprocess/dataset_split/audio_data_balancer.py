import os
import random
import shutil

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.preprocess.audio_waveform import preprocess


random.seed(42)


def _load_waveform(file_path: str) -> np.ndarray:
    y_waveform, sampling_rate_waveform = librosa.load(file_path, sr=32000)

    return y_waveform, sampling_rate_waveform


class AudioDataBalancer:
    """
    Class to balance audio data.

    Parameters
    ----------
    target_sample_size: int, target sample size each class should have after balancing.
    target_wav_duration: float, target duration of each audio sample.
    filepath_orig_data: str, file path to the original, unbalanced data.
    target_dir_balanced_data: str, path to the directory where the balanced data should be stored.
    target_waveform_duration: float, duration each class sample should have.

    Attributes
    ----------
    class_dist: dict, the dictionary which holds the initial samples per class.
    """
    def __init__(self,
                 target_sample_size: int = None,
                 target_wav_duration: float = None,
                 filepath_orig_data: str = None,
                 target_dir_balanced_data: str = None,
                 target_waveform_duration: float = None
                 ):
        self.target_sample_size = target_sample_size
        self.target_wav_duration = target_wav_duration
        self.filepath_orig_data = filepath_orig_data
        self.target_dir_balanced_data = target_dir_balanced_data
        self.target_waveform_duration = target_waveform_duration
        self.output_class_distribution = {"primary_label": [],
                                          "filename": []}


    def _create_dirs_for_balanced_data(self):
        """
        If not existing creates the overall target dir and in this target dir the directories
        for each of the bird classes.
        """
        if not os.path.exists(self.target_dir_balanced_data):
            os.makedirs(self.target_dir_balanced_data)
        for key in self.class_dist:
            class_dir = os.path.join(DATASET_BASE_FILE_PATH + TARGET_DIR + "\\" + str(key))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)


    def _create_balanced_data_distribution_csv(self):
        """
        Creates a datafraem from the output_class_distribution dictionary and
        writes it to a csv file.
        """
        output_classes_dist_df = pd.DataFrame(data=self.output_class_distribution)
        csv_filepath = DATASET_BASE_FILE_PATH + "\\output_classes_distribution.csv"
        output_classes_dist_df.to_csv(csv_filepath, index=False)


    def _preprocess_from_oversampled_classes(self, single_class_name: str):
        """
        :param single_class_name: str, the bird class for which the audio samples should be copied to
        the respective target directory.

        If the bird class is oversampled the first self.target_sample_size samples are truncated or filled
        and then copied to the respective target directoy.
        """
        source_dir_class = DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR + "\\" + str(single_class_name)
        target_dir_class = DATASET_BASE_FILE_PATH + TARGET_DIR + "\\" + str(single_class_name)

        source_files = os.listdir(source_dir_class)
        for file_indice in range(0, self.target_sample_size):
            wav, samplerate = _load_waveform(source_dir_class + "\\" + source_files[file_indice])
            processed_wav = preprocess.truncate_or_fill(wav, samplerate, self.target_wav_duration)
            sf.write(target_dir_class + "\\" + source_files[file_indice], processed_wav, samplerate, format='ogg', subtype='vorbis')
            self.output_class_distribution["primary_label"].append(single_class_name)
            self.output_class_distribution["filename"].append(single_class_name + "/" + source_files[file_indice])


    def _preprocess_for_undersampled_classes(self, single_class_name: str):
        """
        :param single_class_name: str, the bird class for which the audio samples should be copied to
        the respective target directory.

        Copies existing waveforms after truncation or filling to the target directory and then uses augmentation
        techniques to sample up the undersampled classes.
        """
        augmented_filename_waveform_sample_rate = {}
        orig_filename_waveform_sample_rate = {}
        source_dir_class = DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR + "\\" + str(single_class_name)
        target_dir_class = DATASET_BASE_FILE_PATH + TARGET_DIR + "\\" + str(single_class_name)

        source_files = os.listdir(source_dir_class)
        missing_samples = self.target_sample_size - len(source_files)
        for file_indice in range(0, len(source_files)):
            wav, samplerate = _load_waveform(source_dir_class + "\\" + source_files[file_indice])
            processed_wav = preprocess.truncate_or_fill(wav, samplerate, self.target_wav_duration)
            filename = target_dir_class + "\\" + source_files[file_indice].split('.')[0]
            orig_filename_waveform_sample_rate[filename] = [wav, samplerate, processed_wav]

        while missing_samples > 0:
            for file in orig_filename_waveform_sample_rate:
                # TODO: Augmentation following some strategy, not only random
                random_aug_method = random.choice(list(preprocess.available_functions))
                processed_wav = preprocess.available_functions[random_aug_method](orig_filename_waveform_sample_rate[file][0], 
                                                                                        orig_filename_waveform_sample_rate[file][1])
                processed_andtruncated_wav = preprocess.truncate_or_fill(processed_wav, samplerate, self.target_wav_duration)
                
                aug_filename = file + "_" + random_aug_method + "_" + str(missing_samples)
                augmented_filename_waveform_sample_rate[aug_filename] = [orig_filename_waveform_sample_rate[file][0], 
                                                                            orig_filename_waveform_sample_rate[file][1], 
                                                                            processed_andtruncated_wav]
                
                missing_samples -= 1

        for filename in orig_filename_waveform_sample_rate:
            sf.write(filename + '.ogg', orig_filename_waveform_sample_rate[filename][2], 
                     orig_filename_waveform_sample_rate[filename][1], format='ogg', subtype='vorbis')
            self.output_class_distribution["primary_label"].append(single_class_name)
            self.output_class_distribution["filename"].append(single_class_name + "/" + filename.split("\\")[-1] + '.ogg')

        for filename in augmented_filename_waveform_sample_rate:
            sf.write(filename + '.ogg', augmented_filename_waveform_sample_rate[filename][2], 
                    augmented_filename_waveform_sample_rate[filename][1], format='ogg', subtype='vorbis')
            self.output_class_distribution["primary_label"].append(single_class_name)
            self.output_class_distribution["filename"].append(single_class_name + "/" + filename.split("\\")[-1] + '.ogg')


    def balance_dataset(self, direct_class_dist: dict=None):
        """
        :param direct_class_dist: dict, the dictionary holding the number of samples for each class.

        For each bird class do the balancing by either use the waveforms of oversampled classes directly 
        or oversample undersampled classes
        """
        if direct_class_dist:
            self.class_dist = direct_class_dist
        self._create_dirs_for_balanced_data()
        
        key_list = list(self.class_dist.keys())
        for index, bird_class in enumerate(key_list):
            print(bird_class, index, len(self.class_dist))
            if self.class_dist[bird_class] >= self.target_sample_size:
                self._preprocess_from_oversampled_classes(bird_class)
            else:
                self._preprocess_for_undersampled_classes(bird_class)

        self._create_balanced_data_distribution_csv()



    def get_class_dist_from_metadata(self, metadata_filepath: str, class_column: str):
        """
        :param metadata_filepath: str, filepath to the metadata file.
        :param class_column: str, specifies the column where the classes are stored in.

        Extract the class distribution from a metadata csv-file.
        """
        intermediate_class_dist = {}
        # currently only csv metadata files supported
        metadata_file = pd.read_csv(metadata_filepath)
        df = metadata_file[class_column].value_counts().to_frame().reset_index().rename(columns={'index':'class_name', 'values':'count'})
        df_dict_list = df.to_dict(orient='records')
        for key_value_pair in df_dict_list:
            intermediate_class_dist[key_value_pair['class_name']] = key_value_pair[class_column]
        
        self.class_dist = intermediate_class_dist


if __name__ == "__main__":
    # example on how to use it
    DATASET_BASE_FILE_PATH = r"D:\Datasets\birdclef-2023"
    TRAIN_SET_FILE_DIR = r"\train_audio"
    TARGET_DIR = r"\train_audio_balanced"
    metadata_file = os.path.join(DATASET_BASE_FILE_PATH + r"\train_metadata.csv")

    audio_balancer = AudioDataBalancer(target_sample_size=50,
                                       target_wav_duration=5,
                                       target_dir_balanced_data = os.path.join(DATASET_BASE_FILE_PATH + TARGET_DIR)
                                       )
    audio_balancer.get_class_dist_from_metadata(metadata_file, "primary_label")
    audio_balancer.balance_dataset()
