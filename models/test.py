
import torch
from transformers import AutoModelForAudioClassification, Trainer, TrainingArguments

# load the trained model for prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForAudioClassification.from_pretrained("/content/drive/MyDrive/Colab Notebooks/dementia/fine_tuned_model")
model=model.to(device)
trainer = Trainer(model)

#pre-processing the test set 
processed_dataset = dataset.map(preprocess_function, batched=True, writer_batch_size=2, batch_size=2)

# make predictions of test set
predictions, _, _=trainer.predict(processed_dataset)

# get the label
final_prediction = np.argmax(predictions, axis=-1)

# the performance on the test set
from sklearn.metrics import roc_auc_score
roc_auc_score(y, predictions[:,1])