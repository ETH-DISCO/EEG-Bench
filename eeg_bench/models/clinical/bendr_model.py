from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal, Optional
import numpy as np
from .BENDR import utils
import torch
import random
import numpy as np
from mne.io import BaseRaw
import torch.nn as nn
import torch
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from .BENDR.dn3_ext import ConvEncoderBENDR
from ...config import get_config_value
from collections import Counter
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse, make_multilabels
import gc
import os
from pathlib import Path
import requests

def check_and_download_encoder():
    chkpt_dir = Path(get_config_value("chkpt"))
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir, exist_ok=True)
    encoder_path = chkpt_dir / "encoder.pt"
    if not os.path.exists(encoder_path):
        print("Encoder file not found. Downloading encoder.pt ...")
        url = "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt"
        response = requests.get(url, stream=True)
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        with open(encoder_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return encoder_path

class BENDRBCIModel(nn.Module):
    def __init__(self, num_classes, num_labels_per_chunk, chunks):
        super().__init__()
        self.chunks = chunks
        encoder = ConvEncoderBENDR(20, encoder_h=512, dropout=0., projection_head=False)
        encoder.load(check_and_download_encoder())

        self.model = encoder
        for param in self.model.parameters():
            param.requires_grad = False
        self.chunk_length = 1600 # for PD it was 800 (with 4608)
        self.scale_param    = torch.nn.Parameter(torch.tensor(1.))
        self.is_multilabel_task = num_labels_per_chunk is not None
        multiplier = 2 if self.is_multilabel_task else 1
        self.linear_probe   = torch.nn.Linear(multiplier * 8704, num_classes *(num_labels_per_chunk if self.is_multilabel_task else 1))
        self.dropout           = torch.nn.Dropout(p=0.10)
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, n_channels, n_timepoints)
        """

        if self.chunks is not None and (self.chunks == 60 or self.is_multilabel_task):
            x = torch.cat([x, self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))], dim=-2)
        
            h = self.model(x)
            h = h.flatten(1)
            #h = self.dropout(h)

            logits = self.linear_probe(h)
            if self.is_multilabel_task:
                # for multilabel classification, pytorch Cross-Entropy loss expects this prediction shape: [#batch, #classes, #labels]
                logits = logits.reshape((logits.shape[0], self.num_classes, -1))
            return logits

        B, C, T = x.shape
        n_chunks = T // self.chunk_length
        x = x[:, :, :n_chunks * self.chunk_length]
        
        # Reshape into segments:
        x = x.view(B, C, n_chunks, self.chunk_length)
        # Permute to (batch_size, num_chunks, n_channels, chunk_length)
        x = x.permute(0, 2, 1, 3)
        # Merge batch and chunk dimensions for efficient processing:
        x = x.reshape(B * n_chunks, C, self.chunk_length)
        
        scale_param = self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))
        x = torch.cat([x, scale_param], dim=-2)
        
        # Pass through the encoder:
        embeddings = self.model(x)
        embeddings = embeddings.flatten(1)
        
        # Restore the batch and chunk dimensions:
        embedding_dim = embeddings.shape[1]
        embeddings = embeddings.view(B, n_chunks, embedding_dim)
        
        # Simple aggregation: mean pooling over segments
        aggregated = embeddings.mean(dim=1)
        #aggregated = self.dropout(aggregated)
        
        logits = self.linear_probe(aggregated)
        return logits

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0, 0

    for x, y, channels in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)  # batch size is 1
        logits = model(x)
        loss = model.loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        running_corrects += torch.sum(preds == y).item()
        total_samples += x.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for x, y, channels in tqdm(dataloader, desc="Validation"):
            logits = model.forward(x.to(device))
            y = y.to(device)
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
    if model.is_multilabel_task:
        # additionally flatten, i.e. reduce multilabel to single-classification task
        all_labels = all_labels.flatten()
        all_logits = all_logits.transpose(-1, -2).flatten(0, 1)
    
    # Compute additional metrics using get_metrics
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
    results = utils.get_metrics(all_logits.numpy(), all_labels.numpy(), metrics, False)
    
    return epoch_loss, epoch_acc, results


def inference(model, dataloader, device):
    model.eval()
    predictions = []
    indices = []
    with torch.no_grad():
        for x, idx, channels in tqdm(dataloader, desc="Testing"):
            logits = model.forward(x.to(device))
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred.cpu())
            indices.append(idx)
    predictions = torch.cat(predictions, dim=0).cpu()
    indices = torch.cat(indices, dim=0).cpu()
    return predictions, indices


class BENDRModel(AbstractModel):
    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None
    ):
        super().__init__("BENDRModel")
        print("inside init of BENDRModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.chunk_len_s = None if num_labels_per_chunk is None else 16
        self.use_cache = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels_per_chunk = num_labels_per_chunk
        self.model = BENDRBCIModel(num_classes=num_classes, num_labels_per_chunk=num_labels_per_chunk, chunks=self.chunk_len_s).to(self.device)

    def fit(self, X: List[np.ndarray|List[BaseRaw]], y: List[np.ndarray|List[str]], meta: List[Dict]) -> None:
        print("inside fit")
        task_name = meta[0]["task_name"]
        
        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        print("class_weights", class_weights)
        self.model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        dataset_train  = make_dataset_2(X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=self.use_cache)

        val_split = 0.2
        if val_split is not None:
            dataset_train, dataset_val = dataset_train.split_train_val(val_split)
        else:
            dataset_val = None

        del X, y, meta
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.chunk_len_s is None:
            batch_size = 1
        else: 
            batch_size = 64        
        train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=0, shuffle=True)
        if dataset_val is not None:
            valid_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, shuffle=False)
        else:
            valid_loader = None

        max_epochs = 50
        steps_per_epoch = len(train_loader)
        max_lr = 4e-4
        
        # Set up optimizer and OneCycleLR scheduler
        optimizer = torch.optim.AdamW(
            list([self.model.scale_param])+
            list(self.model.model.parameters())+
            list(self.model.linear_probe.parameters()),
            lr=1e-5,
            weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs)

        best_val_loss = float('inf')
        best_model_state = None
        
        # Training loop
        for epoch in range(1, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs}")
            train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, scheduler, self.device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}")

            if valid_loader is not None:
                val_loss, val_acc, val_metrics = validate_epoch(self.model, valid_loader, self.device)
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print("  Val Metrics:", val_metrics)
            
                # Optionally save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
        
        # Load the best model (if saved)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
    

    @torch.no_grad()
    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        task_name = meta[0]["task_name"]
        dataset_test  = make_dataset_2(X, None, meta, task_name, self.name, self.chunk_len_s, is_train=False, use_cache=self.use_cache)

        if len(dataset_test) == 0:
            return np.array([])
        
        # Inference on test set

        if self.chunk_len_s is None:
            batch_size = 1
        else: 
            batch_size = 64
        test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=0, shuffle=False)

        predictions, indices_mapping = inference(self.model, test_loader, self.device)
        
        predictions = predictions.numpy()
        indices_mapping = indices_mapping.numpy()
        print(predictions.shape)
        print(indices_mapping.shape)

        if self.chunk_len_s is not None and not self.model.is_multilabel_task:
            # Aggregate predictions by majority voting for each unique index
            unique_indices = np.unique(indices_mapping)
            aggregated_predictions = []

            for idx in unique_indices:
                # Get all predictions corresponding to the current index
                idx_predictions = predictions[indices_mapping == idx]
                # Perform majority voting
                most_common_prediction = Counter(idx_predictions).most_common(1)[0][0]
                aggregated_predictions.append(most_common_prediction)

            # Convert to numpy array
            predictions = np.array(aggregated_predictions)

        mapped_pred = np.array([map_label_reverse(pred, task_name) for pred in predictions])
       
        print(mapped_pred)
        return mapped_pred
        
