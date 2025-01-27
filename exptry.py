from ultralytics.models.yolo import YOLO
from ultralytics import YOLO
from ultralytics.engine.model import Model
import torch
import wandb
import os
import argparse
# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project="yolov8_softshare")
# Load the custom model configuration
model = YOLO('yolov8n-LD-P2.yaml')
model.model.to(device)

# Define a callback to log losses at the end of each training batch
def log_losses(trainer):
    # Access the loss dictionary
    loss_items = trainer.loss_items
    
    # Log each loss component
    wandb.log({
        "train/box_loss": loss_items[0],
        "train/cls_loss": loss_items[1],
        "train/dfl_loss": loss_items[2]
    }, step=trainer.epoch)

    torch.cuda.empty_cache()

# Register the callback with the YOLO model
model.add_callback('on_train_batch_end', log_losses)

# Train the model with the specified configuration and sync to W&B
Result_Final_model = model.train(
    data="/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml",
    epochs=70,
    batch=16,
    imgsz = 640,
    optimizer='auto',
    project='yolov8_softshare',
    save=True,
)
# Define model and dataset names
model_name = "yolov8_spd"
dataset_name = "waid_yolov8"

# Save the model as .pth file in Kaggle workspace
save_path = f"/kaggle/working/models/{model_name}_{dataset_name}.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.model.state_dict(), save_path)
torch.cuda.empty_cache()

# Finish the W&B run
wandb.finish()
