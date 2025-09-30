from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal, Optional
import numpy as np
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse, LaBraMDataset2, make_multilabels
from .LaBraM import utils
import torch
from timm.models import create_model
import numpy as np
from mne.io import BaseRaw
from .LaBraM import modeling_finetune # important to load the models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from ...config import get_config_value
import gc
from collections import Counter
import logging
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
    def __init__(self, num_classes, num_labels_per_chunk, device, chunks):
        super().__init__()
        self.device = device
        self.chunks = chunks
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
                p.requires_grad = False
        self.feature = model
        self.is_multilabel_task = num_labels_per_chunk is not None
        self.head = nn.Linear(200, num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1))
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, input_chans):
        B, C, T = x.shape

        if self.chunks is not None and (self.chunks <= 10 or self.is_multilabel_task):
            x = x.to(self.device)
            if T % 200 != 0: 
                x = x[:,:,0:T-T%200]
                T = T - T % 200
            x = x.reshape((B, C, T // 200, 200))
            x = x / 100
            
            pred = self.feature.forward_features(x, input_chans=input_chans, return_all_tokens=False)

            pred = self.head(pred.flatten(1))
            if self.is_multilabel_task:
                # for multilabel classification, pytorch Cross-Entropy loss expects this prediction shape: [#batch, #classes, #labels]
                pred = pred.reshape((B, self.num_classes, -1))
            return x, pred

        if len(input_chans) <= 24:
            chunk_length = 2000
        elif len(input_chans) <= 32:
            chunk_length = 1600
        elif len(input_chans) <= 50:
            chunk_length = 1000
        elif len(input_chans) <= 64:
            chunk_length = 800
        else:
            raise ValueError("Unsupported input channel configuration: {}".format(input_chans))

        n_chunks = T // chunk_length
        if n_chunks < 1:
            raise ValueError(
                "Recording too short: expected at least one chunk of length {}, got T={}".format(chunk_length, T)
            )
        # Crop extra samples to have only full chunks
        T_new = n_chunks * chunk_length
        x = x[:, :, :T_new]  # shape: (B, C, T_new)

        # Reshape to split recording into chunks:
        x = x.reshape(B, C, n_chunks, chunk_length)
        x = x.permute(0, 2, 1, 3)  # shape: (B, n_chunks, C, chunk_length)

        # Merge batch and chunks dimensions to process all chunks together:
        x = x.reshape(B * n_chunks, C, chunk_length)
        
        # Tokenize each chunk: each token is 200 samples.
        tokens = x.reshape(B * n_chunks, C, chunk_length // 200, 200)
        tokens = tokens / 100.0

        tokens = tokens.to(self.device)

        # Extract features for each chunk using the pre-trained feature extractor.
        # Expected output shape: (B * n_chunks, feature_dim)
        chunk_features = self.feature.forward_features(tokens, input_chans=input_chans, return_all_tokens=False)
        feature_dim = chunk_features.shape[-1]
        
        # Reshape back to separate recordings and chunks: (B, n_chunks, feature_dim)
        chunk_features = chunk_features.view(B, n_chunks, feature_dim)
        
        # Aggregate features across chunks by averaging (mean pooling)
        aggregated_features = chunk_features.mean(dim=1)  # shape: (B, feature_dim)

        # Get the recording-level prediction from the head.
        logits = self.head(aggregated_features)
        
        return aggregated_features, logits

def train_epoch(model, dataloader, optimizer, scheduler, device, input_chans):
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0, 0

    for batch in tqdm(dataloader, desc="Training", leave=True):
        x, y, channels = batch
        # x = x.to(device) will be done in the model
        y = y.to(device)
        
        if channels != -1 and channels[0] != -1:
            channels = [ch_arr[0] for ch_arr in channels]
            input_chans = utils.get_input_chans(channels)
        
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(x, input_chans)
        loss = model.loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        running_corrects += torch.sum(preds == y).item()
        total_samples += x.size(0)

        del x, y, logits, loss  # Delete tensors no longer needed
        gc.collect()  # Invoke garbage collection
        torch.cuda.empty_cache()  # Clear cached memory on GPU
        
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
        for batch in tqdm(dataloader, desc="Validation", leave=True):
            x, y, channels = batch
            #x = x.to(device) will be done in the model
            y = y.to(device)
            
            if channels != -1 and channels[0] != -1:
                channels = [ch_arr[0] for ch_arr in channels]
                input_chans = utils.get_input_chans(channels)
            
            _, logits = model(x, input_chans)
            loss = model.loss_fn(logits, y)
            
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += torch.sum(preds == y).item()
            total_samples += x.size(0)
            
            all_labels.append(y.cpu())
            all_logits.append(logits.cpu())

            del x, y, logits  # Delete tensors no longer needed
            torch.cuda.empty_cache()  # Clear cached memory on GPU
    
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

def inference(model, dataloader, device, input_chans):
    model.eval()
    predictions = []
    indices = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=True):
            x, idx, channels  = batch
            if channels != -1 and channels[0] != -1:
                channels = [ch_arr[0] for ch_arr in channels]
                input_chans = utils.get_input_chans(channels)

            # x = x.to(device) will be done in the model
            _, logits = model(x, input_chans)
            preds = torch.argmax(logits, dim=1)
            predictions.append(preds.cpu())
            indices.append(idx)

            del x, idx, logits  # Delete tensors no longer needed
            torch.cuda.empty_cache()  # Clear cached memory on GPU
    predictions = torch.cat(predictions, dim=0).cpu()
    indices = torch.cat(indices, dim=0).cpu()
    return predictions, indices

class LaBraMModel(AbstractModel):
    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
    ):
        super().__init__("LaBraMModel")
        print("inside init LaBraMModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.chunk_len_s = None if num_labels_per_chunk is None else 16
        self.use_cache = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels_per_chunk = num_labels_per_chunk
        self.model = LaBraMBCIModel(num_classes=num_classes, num_labels_per_chunk=num_labels_per_chunk, device=self.device, chunks=self.chunk_len_s).to(self.device)
        self.save = False

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

        ch_names_train = dataset_train.ch_names
        if dataset_val is not None:
            ch_names_val = dataset_val.ch_names


        if self.chunk_len_s is None:
            batch_size = 1
        else: 
            batch_size = 64
        train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
        if dataset_val is not None:
            valid_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)
        else:
            valid_loader = None

        max_epochs = 30
        steps_per_epoch = len(train_loader)
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

        start_epoch = 1
        if self.save and os.path.exists(os.path.join(get_config_value("chkpt"), "labram_checkpoint.pth")):
            checkpoint = torch.load(os.path.join(get_config_value("chkpt"), "labram_checkpoint.pth"))
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_loss"]
            best_model_state = checkpoint["best_model"]
            print(f"Resuming training at epoch {start_epoch}")

        # Training loop
        for epoch in range(start_epoch, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs}")
            input_chans = utils.get_input_chans(ch_names_train)
            train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, scheduler, self.device, input_chans)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}")

            if valid_loader is not None:
                input_chans = utils.get_input_chans(ch_names_val)
                val_loss, val_acc, val_metrics = validate_epoch(self.model, valid_loader, self.device, input_chans)
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print("  Val Metrics:", val_metrics)
        
                # Optionally save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
            
            if self.save:
                torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_model": best_model_state,
                    "best_loss": best_val_loss,
                }, os.path.join(get_config_value("chkpt"), "labram_checkpoint.pth"))
        
        # Load the best model (if saved)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    @torch.no_grad()
    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        task_name = meta[0]["task_name"]
        dataset_test  = make_dataset_2(X, None, meta, task_name, self.name, self.chunk_len_s, is_train=False, use_cache=self.use_cache)
        ch_names = dataset_test.ch_names
        
        if len(dataset_test) == 0:
            return np.array([])
        
        # Inference on test set

        if self.chunk_len_s is None:
            batch_size = 1
        else: 
            batch_size = 64
        test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)

        input_chans = utils.get_input_chans(ch_names)
        predictions, indices_mapping = inference(self.model, test_loader, self.device, input_chans)
        
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
        
