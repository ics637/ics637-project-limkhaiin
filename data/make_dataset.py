
## load and pre-processing 
## import training data
training_data=load_dataset('/content/drive/MyDrive/Colab Notebooks/dementia/English/Pitt', name="en-US",  drop_labels=False)
training_data = training_data.cast_column("audio", Audio(sampling_rate=16_000))

## load and pre-processing test set
dataset_test = load_dataset('/content/drive/MyDrive/Colab Notebooks/dementia/English/ADReSS-2021/audio/', name="en-US", split='train')
dataset_test = dataset.cast_column("audio", Audio(sampling_rate=16000))