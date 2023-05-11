
!pip install transformers datasets evaluate

from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback
from transformers.optimization import Adafactor, AdafactorSchedule
from copy import deepcopy

## Override loss function with binary cross-entropy loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.BCELoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

## Using AUROC as the metric
roc_auc = evaluate.load("roc_auc")
def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return roc_auc.compute(prediction_scores=predictions, references=eval_pred.label_ids)

## Loadoad the pre-trained wav2vec model
#num_labels = len(id2label)
num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
   "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label)


## disable the gradient computation for the feature encoder so that its parameter will not be updated during training.
model.freeze_feature_encoder()

## hyperparameters of the model
training_args = TrainingArguments(
       evaluation_strategy="epoch",
       output_dir="/content/drive/MyDrive/Colab Notebooks/dementia",
       save_strategy="epoch",
       learning_rate=1e-5,
       per_device_train_batch_size=32,
       gradient_accumulation_steps=1,
       per_device_eval_batch_size=32,
       num_train_epochs=30,
       warmup_ratio=0.1,
       logging_steps=10,
       weight_decay=0.01,
       load_best_model_at_end=True,
       metric_for_best_model="roc_auc",
 )
optimizer = Adafactor(model.parameters(), lr=1e-7, relative_step=False, warmup_init=False)
lr_scheduler = AdafactorSchedule(optimizer)
optimizers = optimizer, lr_scheduler
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_final_data['train'],
        eval_dataset=processed_final_data["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
        )

## an override function which outputs the training auroc scores
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

## training
trainer.add_callback(CustomCallback(trainer)) 
trainer.train()