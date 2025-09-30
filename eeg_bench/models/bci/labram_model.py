from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal
import numpy as np
from .LaBraM.make_dataset import make_dataset
from .LaBraM.utils_2 import calc_class_weights, reverse_map_label, n_unique_labels
from .LaBraM import utils
import torch
from timm.models import create_model
import random
import numpy as np
import datetime
from mne.io import BaseRaw
from .LaBraM import modeling_finetune # important to load the models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from joblib import Memory
from ...config import get_config_value
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import os
import requests
from pathlib import Path

def check_and_download_pretrained_model():
    chkpt_dir = Path(get_config_value("chkpt"))
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir, exist_ok=True)
    encoder_path = chkpt_dir / "labram-base.pth"
    if not os.path.exists(encoder_path):
        print("Labram-Base file not found. Downloading labram-base.pth ...")
        url = "https://github.com/935963004/LaBraM/raw/refs/heads/main/checkpoints/labram-base.pth"
        response = requests.get(url, stream=True)
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        with open(encoder_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return encoder_path

class LaBraMBCIModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        checkpoint = torch.load(check_and_download_pretrained_model())
        new_checkpoint = {}
        for k,v in checkpoint['model'].items():
            if k.startswith('student.'):
                new_checkpoint[k[len('student.'):]] = v
        model = create_model("labram_base_patch200_200", 
                                # checkpoint_path= ,
                                qkv_bias=False,
                                rel_pos_bias=True,
                                num_classes=num_classes,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                attn_drop_rate=0.0,
                                drop_block_rate=None,
                                use_mean_pooling=True,
                                init_scale=0.001,
                                use_rel_pos_bias=True,
                                use_abs_pos_emb=True,
                                init_values=0.1,)
        model.load_state_dict(new_checkpoint, strict=False)
        for blk in model.blocks:
            for p in blk.parameters():
                p.requires_grad = True
        self.feature = model
        self.head = nn.Linear(200, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, input_chans):
        B, C, T = x.shape
        if T % 200 != 0: 
            x = x[:,:,0:T-T%200]
            T = T - T % 200
        x = x.reshape((B, C, T // 200, 200))
        x = x / 100
        
        pred = self.feature.forward_features(x, input_chans=input_chans, return_all_tokens=False)

        pred = self.head(pred.flatten(1))
        return x, pred

def train_epoch(model, dataloader, optimizer, scheduler, device, input_chans):
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        x, y = batch
        x = x.to(device)
        y = y.to(device).argmax(dim=1)
        
        optimizer.zero_grad()
        _, logits = model(x, input_chans)
        loss = model.loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()  # update the learning rate
        
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        running_corrects += torch.sum(preds == y).item()
        total_samples += x.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, device, input_chans):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            x, y = batch
            x = x.to(device)
            y = y.to(device).argmax(dim=1)
            
            _, logits = model(x, input_chans)
            loss = model.loss_fn(logits, y)
            
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += torch.sum(preds == y).item()
            total_samples += x.size(0)
            
            all_labels.append(y.cpu())
            all_logits.append(logits.cpu())
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    # Concatenate predictions and labels
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Compute additional metrics using get_metrics
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
    results = utils.get_metrics(all_logits.numpy(), all_labels.numpy(), metrics, False)
    
    return epoch_loss, epoch_acc, results

def inference(model, dataloader, device, input_chans):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in tqdm(dataloader, desc="Testing", leave=False):
            x = x.to(device)
            _, logits = model(x, input_chans)
            preds = torch.argmax(logits, dim=1)
            predictions.append(preds.cpu())
    predictions = torch.cat(predictions, dim=0)
    return predictions


class LaBraMModel(AbstractModel):
    def __init__(
        self,
    ):
        super().__init__("LaBraMModel")
        print("inside init of LaBraMModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = Memory(location=get_config_value("cache"), verbose=0)


    def fit(self, X: List[np.ndarray|List[BaseRaw]], y: List[np.ndarray|List[str]], meta: List[Dict]) -> None:
        print("inside fit of LaBraMModel")
        logging.info("inside fit of LaBraMModel")
        task_name = meta[0]["task_name"]

        num_classes = n_unique_labels(task_name)
        self.model = LaBraMBCIModel(num_classes=num_classes).to(self.device)

        datasets = [self.cache.cache(make_dataset)(X_, y_, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=True, split_size=0.15)
                for X_, y_, meta_ in tqdm(zip(cast(List[np.ndarray], X), cast(List[np.ndarray], y), meta),
                              desc="Creating datasets", total=len(meta))]
        dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_val_list = [dataset[1] for dataset in datasets]

        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]        
        ch_names_list_train = [dataset.ch_names for dataset in dataset_train_list]
        if dataset_val_list is not None:
            dataset_val_list = [dataset for dataset in dataset_val_list if len(dataset) > 0]
            ch_names_list_val = [dataset.ch_names for dataset in dataset_val_list]

        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        print("class_weights", class_weights)
        self.model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        del X, y, meta

        torch.cuda.empty_cache()

        batch_size = 64
        train_loader_list = [DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True) for train_dataset in dataset_train_list]
        valid_loader_list = [DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for valid_dataset in dataset_val_list]
        
        max_epochs = 50
        steps_per_epoch = math.ceil(sum([len(train_loader) for train_loader in train_loader_list]))
        max_lr = 4e-4
        
        
        # Set up optimizer and OneCycleLR scheduler
        optimizer = torch.optim.AdamW(
            list(self.model.head.parameters()) + 
            list(self.model.feature.parameters()), 
            lr=1e-6, 
            weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        

        best_val_loss = float('inf')
        best_model_state = None
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Training loop
        for epoch in range(1, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs} with LR: {scheduler.get_last_lr()}")

            epoch_train_loss = 0
            epoch_train_acc = 0
            num_train_batches = 0
            
            train_pairs = list(zip(train_loader_list, ch_names_list_train))
            random.shuffle(train_pairs)
            for train_loader, ch_names in train_pairs:
                input_chans = utils.get_input_chans(ch_names)
                train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, scheduler, self.device, input_chans)
                epoch_train_loss += train_loss
                epoch_train_acc += train_acc
                num_train_batches += 1
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

            # Averaging over the training batches
            avg_train_loss = epoch_train_loss / num_train_batches
            avg_train_acc = epoch_train_acc / num_train_batches
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            
            epoch_val_loss = 0
            epoch_val_acc = 0
            num_val_batches = 0
            
            for valid_loader, ch_names in zip(valid_loader_list, ch_names_list_val):
                input_chans = utils.get_input_chans(ch_names)
                val_loss, val_acc, val_metrics = validate_epoch(self.model, valid_loader, self.device, input_chans)
                epoch_val_loss += val_loss
                epoch_val_acc += val_acc
                num_val_batches += 1
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print("  Val Metrics:", val_metrics)
            
            # Averaging over the validation batches
            avg_val_loss = epoch_val_loss / num_val_batches
            avg_val_acc = epoch_val_acc / num_val_batches
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)
            
            # Optionally save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
        
        # Load the best model (if saved)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
    

    @torch.no_grad()
    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        task_name = meta[0]["task_name"]
        dataset_test_list = [self.cache.cache(make_dataset)(X_, None, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=False, split_size=0) 
                             for X_, meta_ in tqdm(zip(cast(List[np.ndarray], X), meta),
                             desc="Creating datasets", total=len(meta))]
        dataset_test_list = [dataset for dataset in dataset_test_list if len(dataset) > 0]
        print("datasets length: ", len(dataset_test_list[0]))
        ch_names_list = [dataset.ch_names for dataset in dataset_test_list]
        # Inference on test set

        batch_size = 64
        test_loader_list = [DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for test_dataset in dataset_test_list]

        predictions = []
        for test_loader, ch_names in zip(test_loader_list, ch_names_list):
            input_chans = utils.get_input_chans(ch_names)
            predictions.append(inference(self.model, test_loader, self.device, input_chans).cpu())
        
        predictions = torch.cat(predictions, dim=0).numpy()

        mapped_pred = np.array([reverse_map_label(idx, task_name) for idx in predictions])
        
        print(mapped_pred)
        return mapped_pred
        
