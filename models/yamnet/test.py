
# Load the model
my_model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/dementia/tensorflow_model.h5')

# load the test set
control_test_filenames=glob.glob('/content/drive/MyDrive/Colab Notebooks/dementia/English/ADReSS-2021/audio/control/*.wav')
control_test_labels=(len(control_test_filenames))*[0]
dementia_test_filenames=glob.glob('/content/drive/MyDrive/Colab Notebooks/dementia/English/ADReSS-2021/audio/dementia/*.wav')
dementia_test_labels=(len(dementia_test_filenames))*[1]

# preprocess the test set
all_test_folds = [6]*1444
all_test_filenames=control_test_filenames+dementia_test_filenames
all_test_labels=control_test_labels+dementia_test_labels
test_ds=tf.data.Dataset.from_tensor_slices((all_test_filenames, all_test_labels, all_test_folds))
test_ds = test_ds.map(load_wav_for_map)
test_ds = test_ds.map(extract_embedding).unbatch()
remove_fold_column = lambda embedding, label, fold: (embedding, label)
test_ds = test_ds.map(remove_fold_column)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

# Evaluate the model using the test set
loss, accuracy = my_model.evaluate(test_ds, verbose=1)
print(loss)
print(accuracy)