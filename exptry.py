from ultralytics import RTDETR
import torch
import wandb
import os

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize a new W&B run
wandb.init(project="rtdetr_exp")

# Load the RT-DETR model configuration
model = RTDETR('rtdetr.yaml')
model.model.to(device)

# Define a callback to log losses at the end of each training batch
def log_losses(trainer):
    loss_items = trainer.loss_items
    wandb.log({
        "train/box_loss": loss_items[0],
        "train/cls_loss": loss_items[1],
        "train/dfl_loss": loss_items[2]
    }, step=trainer.epoch)
    torch.cuda.empty_cache()

# Register the callback with the RT-DETR model
model.add_callback('on_train_batch_end', log_losses)

# Train the model with the specified configuration and sync to W&B
Result_Final_model = model.train(
    data="/content/drive/MyDrive/WAID-main/WAID-main/WAID/data.yaml",
    epochs=70,
    batch=4,
    optimizer='SOAP',
    project='rtdetr_new_exp',
    save=True,
    imgsz=640
)

# Finish the W&B run
wandb.finish()

