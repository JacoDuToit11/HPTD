# Implementation of data processing, training, and testing procedure for HPTD model.

# imports
from typing import Any
import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
from torch.optim import Adam
from torchmetrics.classification import MultilabelF1Score, BinaryF1Score, MultilabelStatScores, BinaryStatScores

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import torch.optim.lr_scheduler as lr_scheduler
import wandb
# wandb.login()
from pytorch_lightning.loggers import WandbLogger

import numpy as np

from torch.utils.data import DataLoader
import datasets
from datasets import Dataset

from transformers.utils import logging
logging.set_verbosity_error()
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

# Global variables
from constants import *

low_resource = False
# low_resource = True

# initialise_embeddings = 'hard'
initialise_embeddings = 'soft'
# initialise_embeddings = 'flat'
# add_prompts_after_hier_prompt = False
add_prompts_after_hier_prompt = True
graph_type = 'GAT'
graph_layers = 1

# dlm_loss = True
dlm_loss = False
dlm_loss_ratios = [1e-2]
if dlm_loss == False:
    dlm_loss_ratios = [1e-3]

# text_mask = True
text_mask = False

template_at_end = True
# template_at_end = False

# class_biases = [False]
class_biases = [True]

# threshold_tuning_methods = ['0.5', 'per_label']
threshold_tuning_methods = ['0.5', 'single', 'per_level', 'per_label']
# threshold_tuning_methods = ['0.5']

val_monitor = 'harmonic_mean_micro_macro_f1'
# val_monitor = 'macro_f1'
# val_monitor = 'micro_f1'
# val_monitor = 'loss'

# gradient_clipping = 1
gradient_clipping = 0

dataset_names = ['WOS', 'RCV1', 'NYT']

hyperparam_tune = False
# hyperparam_tune = True

test = False
# test = True

# desktop = True
desktop = False

# use_lr_scheduler = True
use_lr_scheduler = False

if test:
    dataset_names = ['NYT']
    # threshold_tuning_methods = ['0.5']
    num_train_samples = 100
    num_dev_samples = 100
    num_test_samples = 100
else:
    num_train_samples = 'all'
    num_dev_samples = 'all'
    num_test_samples = 'all'

sweep_name = 'HPTD'

# Hyperparameters
if test:
    num_epochs = 3
    batch_size = [32]
    learning_rate = [1e-4]
    random_seeds = [42]
    bootstrap_samples = 10
else:
    num_epochs = 20
    batch_size = [16]
    if not hyperparam_tune:
        random_seeds = [33, 55, 77]
        if model_type == 'electra':
            if dataset_names[0] == 'WOS':
                learning_rate = [8e-6]
            elif dataset_names[0] == 'RCV1':
                learning_rate = [1.5e-5]
            elif dataset_names[0] == 'NYT':
                random_seeds = [33, 44, 55]
                learning_rate = [2e-5]
        elif model_type == 'deberta':
            if dataset_names[0] == 'WOS':
                learning_rate = [1.5e-5]
            elif dataset_names[0] == 'RCV1':
                learning_rate = [2e-5]
            elif dataset_names[0] == 'NYT':
                learning_rate = [2e-5]
    else:
        random_seeds = [44]
        learning_rate = [8e-6, 1e-5, 1.5e-5, 2e-5]
    bootstrap_samples = 100

class ValCallback(pl.Callback):
    def on_validation_batch_start(self, trainer, pl_module, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if trainer.global_step == 0: 
            wandb.define_metric("0.5_Val_Macro_F1", summary="max")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_step == 0: 
            wandb.define_metric("0.5_Val_Macro_F1", summary="max")
        val_epoch_outputs = torch.cat(pl_module.val_epoch_outputs, dim = 0)
        val_epoch_labels = torch.cat(pl_module.val_epoch_labels, dim = 0)

        if not training:
            global thresholds
            global threshold_tuning_method
            for i in range(len(threshold_tuning_methods)):
                threshold_tuning_method = threshold_tuning_methods[i]
                if threshold_tuning_method == 'single':
                    thresholds = [x/100.0 for x in range(20, 70, 10)]
                elif threshold_tuning_method == 'per_level':
                    thresholds = [[x/100.0 for x in range(20, 70, 10)] for i in range(len(level_num_labels))]
                elif threshold_tuning_method == 'per_label':
                    thresholds = [[x/100.0 for x in range(20, 70, 10)] for i in range(sum(level_num_labels))]
                if threshold_tuning_method == 'single':
                    pl_module.tune_single_threshold_bootstrap(val_epoch_outputs, val_epoch_labels)
                elif threshold_tuning_method == 'per_level':
                    pl_module.tune_threshold_per_level_bootstrap(val_epoch_outputs, val_epoch_labels)
                elif threshold_tuning_method == 'per_label':
                    pl_module.tune_threshold_per_label_bootstrap(val_epoch_outputs, val_epoch_labels)
                if threshold_tuning_method == 'single':
                    pl_module.wandblogger.experiment.config.update({'best_thresholds_single': pl_module.best_threshold})
                elif threshold_tuning_method == 'per_level':
                    pl_module.wandblogger.experiment.config.update({'best_thresholds_level': pl_module.best_threshold_per_level})
                elif threshold_tuning_method == 'per_label':
                    pl_module.wandblogger.experiment.config.update({'best_thresholds_label': pl_module.best_threshold_per_label})

                pl_module.log_metrics(val_epoch_outputs, val_epoch_labels, threshold_tuning_method + '_Val')

        pl_module.log_metrics(val_epoch_outputs, val_epoch_labels, threshold_tuning_method + '_Val')

        pl_module.val_epoch_outputs.clear()
        pl_module.val_epoch_labels.clear()

def Train():
    if test:
        wandb.init(project = "Masters", name = "Test")
        wandb_logger = WandbLogger(project = "Masters", name = "Test")
    else:
        wandb.init(project = "Masters", name = "Run")
        wandb_logger = WandbLogger(project = "Masters", name = "Run")
    if initialise_embeddings == 'hard' or initialise_embeddings == 'flat':
        wandb_logger.experiment.config.update({'data': dataset_name, 'size': size, 'num_train': num_train_samples, 
                                               'num_test': num_test_samples, 'prompts': initialise_embeddings, 'dlm_loss': dlm_loss, 'text_mask': text_mask, 'dlm_loss_ratio': dlm_loss_ratio, 
                                               'model': model_type, 'template_at_end': template_at_end, 'add_prompts_after_hier_prompt': add_prompts_after_hier_prompt, 'random_seed': random_seed, 'class_bias': class_bias, 
                                               'val_monitor': val_monitor, 'frozen': frozen, 'gradient_clipping': gradient_clipping, 'lr_scheduler': use_lr_scheduler, 
                                               'num_soft_prompts': num_soft_prompts, 'random_prompt': random_prompt, 'token_type_setting': token_type_setting, 'add_SEP': add_SEP})
    elif initialise_embeddings == 'soft':
        wandb_logger.experiment.config.update({'data': dataset_name, 'size': size, 'num_train': num_train_samples, 
                                               'num_test': num_test_samples, 'prompts': initialise_embeddings, 'graph_type': graph_type, 
                                               'graph_layers': graph_layers, 'dlm_loss': dlm_loss, 'text_mask': text_mask, 'dlm_loss_ratio': dlm_loss_ratio, 'model': model_type,
                                                'template_at_end': template_at_end, 'add_prompts_after_hier_prompt': add_prompts_after_hier_prompt, 'random_seed': random_seed, 
                                                'class_bias': class_bias, 'val_monitor': val_monitor, 'frozen': frozen, 'gradient_clipping': gradient_clipping, 'lr_scheduler': use_lr_scheduler, 
                                                'num_soft_prompts': num_soft_prompts, 'random_prompt': random_prompt, 'token_type_setting': token_type_setting, 'add_SEP': add_SEP})
    config=wandb.config
    if config.batch_size == 32:
        train_loader = DataLoader(train_dataset, batch_size = 16)
    else:
        train_loader = DataLoader(train_dataset, batch_size = config.batch_size)

    val_loader = DataLoader(val_dataset, batch_size = 32)
    test_loader = DataLoader(test_dataset, batch_size = 64)
    model = torch.load('base_model.pt')
    model.class_bias = class_bias
    lightning_model = LightningModel(model, level_num_labels, config.lr, wandb_logger)
    val_callback = ValCallback()
    wandb.define_metric("val_loss", summary="min")
    if val_monitor == 'harmonic_mean_micro_macro_f1':
        early_stop_callback = EarlyStopping(monitor="0.5_Val harmonic_mean_micro_macro_f1", mode="max", min_delta=0.00, patience=5)
        checkpoint_callback = ModelCheckpoint(monitor="0.5_Val harmonic_mean_micro_macro_f1", mode="max")
    if val_monitor == 'loss':
        early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.00, patience=5, verbose=False)
    elif val_monitor == 'macro_f1':
        early_stop_callback = EarlyStopping(monitor="0.5_Val Macro F1-score", mode="max", min_delta=0.00, patience=5, verbose=False)
        checkpoint_callback = ModelCheckpoint(monitor="0.5_Val Macro F1-score", mode="max")
    elif val_monitor == 'micro_f1':
        early_stop_callback = EarlyStopping(monitor="0.5_Val Micro F1 score", mode="max", min_delta=0.00, patience=5, verbose=False)
        checkpoint_callback = ModelCheckpoint(monitor="0.5_Val Micro F1 score", mode="max")
    accumulate_grad_batches = 1
    if config.batch_size == 32:
        accumulate_grad_batches = 2
    if desktop:
        accumulate_grad_batches = 4
    trainer = pl.Trainer(precision="16-mixed", max_epochs = num_epochs, logger = wandb_logger, deterministic=True, accelerator = accelerator, callbacks=[early_stop_callback, checkpoint_callback, val_callback], accumulate_grad_batches = accumulate_grad_batches, gradient_clip_val=gradient_clipping)
    global training
    training = True
    global threshold_tuning_method
    threshold_tuning_method = '0.5'
    trainer.fit(model = lightning_model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    wandb_logger.experiment.config.update({'best_val_macro_f1': lightning_model.best_val_macro_f1})
    checkpoint_path = checkpoint_callback.best_model_path
    wandb_logger.experiment.config.update({'checkpoint_path': checkpoint_path})
    training = False

    # Tuning thresholds
    trainer.validate(model = lightning_model, dataloaders = val_loader, ckpt_path = "best")
    trainer.test(lightning_model, test_loader, ckpt_path = "best")

class LightningModel(pl.LightningModule):
    def __init__(self, model, level_num_labels, lr, wandblogger):
        super().__init__()
        self.model = model
        self.level_num_labels = level_num_labels
        self.num_labels = sum(level_num_labels)
        self.lr = lr
        self.wandblogger = wandblogger
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.text_loss_fc = nn.BCEWithLogitsLoss()

        self.val_epoch_outputs = []
        self.val_epoch_labels = []
        self.test_epoch_outputs = []
        self.test_epoch_labels = []

        self.best_val_macro_f1 = 0

        self.best_threshold = 0.5
        self.best_threshold_per_level = [0.5] * len(level_num_labels)
        self.best_threshold_per_label = [0.5] * num_labels

    def obtain_output(self, batch, training_data):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        token_type_ids = batch['token_type_ids']
        label_positions = batch['positions']
        text_length = batch['text_length']
        if dlm_loss and training_data:
            label_logits, text_logits, is_replaced = self.model(input_ids, attention_mask, position_ids, token_type_ids, label_positions, text_length, training_data)
            return label_logits, text_logits, is_replaced
        else:
            output = self.model(input_ids, attention_mask, position_ids, token_type_ids, label_positions, text_length, training_data)
            return output

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        if dlm_loss:
            label_logits, text_logits, is_replaced = self.obtain_output(batch, training_data = True)
            label_logits_input_to_loss = torch.special.logit(label_logits, eps=1e-6)
            label_loss = self.loss_fn(label_logits_input_to_loss, labels)
            self.log("train_label_loss", label_loss, prog_bar=True, on_step=False, on_epoch=True)
            text_loss = self.text_loss_fc(text_logits, is_replaced.float())
            self.log("train_text_loss", text_loss, prog_bar=True, on_step=False, on_epoch=True)
            loss =  (1 - dlm_loss_ratio) * label_loss + dlm_loss_ratio * text_loss
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss
        else:
            label_logits = self.obtain_output(batch, training_data = True)
            label_logits_input_to_loss = torch.special.logit(label_logits, eps=1e-6)
            loss = self.loss_fn(label_logits_input_to_loss, labels)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss

    def log_metrics(self, output, labels, phase):
        if threshold_tuning_method == '0.5' or threshold_tuning_method == 'single':
            total_TP, total_FP, total_FN, TP_per_class, FP_per_class, FN_per_class = self.get_stats_single_threshold(output, labels)
        elif threshold_tuning_method == 'per_level':
            total_TP, total_FP, total_FN, TP_per_class, FP_per_class, FN_per_class = self.get_stats_threshold_per_level(output, labels)
        elif threshold_tuning_method == 'per_label':
            total_TP, total_FP, total_FN, TP_per_class, FP_per_class, FN_per_class = self.get_stats_threshold_per_label(output, labels)

        micro_f1 = self.log_micro_metrics(total_TP, total_FP, total_FN, phase)
        macro_f1 = self.log_macro_metrics(TP_per_class, FP_per_class, FN_per_class, phase)

        self.log_harmonic_mean_micro_macro_f1(micro_f1, macro_f1, phase)

    def log_harmonic_mean_micro_macro_f1(self, micro_f1, macro_f1, phase):
        if micro_f1 + macro_f1 > 0:
            harmonic_mean = (2 * micro_f1 * macro_f1)/(micro_f1 + macro_f1)
        else: 
            harmonic_mean = 0
        self.log(phase + ' harmonic_mean_micro_macro_f1', harmonic_mean, on_step=False, on_epoch=True)

    def get_stats_single_threshold(self, output, labels):
        if threshold_tuning_method == '0.5':
            threshold = 0.5
        elif threshold_tuning_method == 'single':
            threshold = self.best_threshold

        stat_scores_metric = MultilabelStatScores(average=None, num_labels = self.num_labels, threshold = threshold).to(self.device)
            
        stat_scores = stat_scores_metric(output, labels)
        TP_per_class = stat_scores[:, 0].tolist()
        FP_per_class = stat_scores[:, 1].tolist()
        FN_per_class = stat_scores[:, 3].tolist()

        total_TP = sum(TP_per_class)
        total_FP = sum(FP_per_class)
        total_FN = sum(FN_per_class)

        return total_TP, total_FP, total_FN, TP_per_class, FP_per_class, FN_per_class

    def get_stats_threshold_per_level(self, output, labels):
        total_TP, total_FP, total_FN = 0, 0, 0
        TP_per_class, FP_per_class, FN_per_class= [], [], []

        prev_levels_num_labels = 0
        for i in range(len(self.level_num_labels)):
            level_output = output[:, prev_levels_num_labels: prev_levels_num_labels + self.level_num_labels[i]]
            level_labels = labels[:, prev_levels_num_labels: prev_levels_num_labels + self.level_num_labels[i]]
            prev_levels_num_labels += self.level_num_labels[i]
            if self.level_num_labels[i] > 1:
                stat_scores_metric = MultilabelStatScores(average=None, num_labels = self.level_num_labels[i], threshold = self.best_threshold_per_level[i]).to(self.device)
            else:
                stat_scores_metric = BinaryStatScores(threshold = self.best_threshold_per_level[i]).to(self.device)
                
            stat_scores = stat_scores_metric(level_output, level_labels)

            if self.level_num_labels[i] > 1:
                level_TP_per_class = stat_scores[:, 0].tolist()
                level_FP_per_class = stat_scores[:, 1].tolist()
                level_FN_per_class = stat_scores[:, 3].tolist()

                TP_per_class += level_TP_per_class
                FP_per_class += level_FP_per_class
                FN_per_class += level_FN_per_class

                level_TP = sum(level_TP_per_class)
                level_FP = sum(level_FP_per_class)
                level_FN = sum(level_FN_per_class)
            else:
                level_TP_per_class = stat_scores[0].item()
                level_FP_per_class = stat_scores[1].item()
                level_FN_per_class = stat_scores[3].item()

                TP_per_class.append(level_TP_per_class)
                FP_per_class.append(level_FP_per_class)
                FN_per_class.append(level_FN_per_class)

                level_TP = level_TP_per_class
                level_FP = level_FP_per_class
                level_FN = level_FN_per_class

            total_TP += level_TP
            total_FP += level_FP
            total_FN += level_FN

        return total_TP, total_FP, total_FN, TP_per_class, FP_per_class, FN_per_class
    
    def get_stats_threshold_per_label(self, output, labels):
        total_TP, total_FP, total_FN = 0, 0, 0
        TP_per_class, FP_per_class, FN_per_class= [], [], []

        for i in range(sum(self.level_num_labels)):
            class_output = output[:, i]
            class_labels = labels[:, i]
            stat_scores_metric = BinaryStatScores(threshold = self.best_threshold_per_label[i]).to(self.device)
                
            stat_scores = stat_scores_metric(class_output, class_labels)
            temp_TP_per_class = stat_scores[0].item()
            temp_FP_per_class = stat_scores[1].item()
            temp_FN_per_class = stat_scores[3].item()

            TP_per_class.append(temp_TP_per_class)
            FP_per_class.append(temp_FP_per_class)
            FN_per_class.append(temp_FN_per_class)

            total_TP += temp_TP_per_class
            total_FP += temp_FP_per_class
            total_FN += temp_FN_per_class

        return total_TP, total_FP, total_FN, TP_per_class, FP_per_class, FN_per_class
    
    def log_micro_metrics(self, total_TP, total_FP, total_FN, phase):
        if total_TP + total_FP > 0:
            micro_precision =  total_TP/(total_TP + total_FP)
        else:
            micro_precision = 0
        if total_TP + total_FN > 0:
            micro_recall =  total_TP/(total_TP + total_FN)
        else:
            micro_recall = 0
        if micro_precision + micro_recall > 0:
            micro_f1 = 2*((micro_precision * micro_recall)/(micro_precision + micro_recall))
        else:
            micro_f1 = 0

        self.log(phase + ' Micro Precision score', micro_precision, on_step=False, on_epoch=True)
        self.log(phase + ' Micro Recall score', micro_recall, on_step=False, on_epoch=True) 
        self.log(phase + ' Micro F1 score', micro_f1, on_step=False, on_epoch=True)

        return micro_f1

    def log_macro_metrics(self, TP_per_class, FP_per_class, FN_per_class, phase):
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        for i in range(len(TP_per_class)):
            if TP_per_class[i] + FP_per_class[i] != 0:
                temp_precision = TP_per_class[i]/(TP_per_class[i] + FP_per_class[i]) 
            else:
                temp_precision = 0
            precision_per_class.append(temp_precision)

            if TP_per_class[i] + FN_per_class[i] != 0:
                temp_recall = TP_per_class[i]/(TP_per_class[i] + FN_per_class[i])
            else:
                temp_recall = 0
            recall_per_class.append(temp_recall)

            if temp_precision + temp_recall > 0:
                temp_f1 = 2*((temp_precision * temp_recall)/(temp_precision + temp_recall))
            else:
                temp_f1 = 0
            f1_per_class.append(temp_f1)
        
        macro_precision_score = torch.mean(torch.Tensor(precision_per_class)).item()
        macro_recall_score = torch.mean(torch.Tensor(recall_per_class)).item()
        macro_f1_score = torch.mean(torch.Tensor(f1_per_class)).item()

        if training:
            if macro_f1_score > self.best_val_macro_f1:
                self.best_val_macro_f1 = macro_f1_score

        self.log(phase + ' Macro Precision score', macro_precision_score)
        self.log(phase + ' Macro Recall score', macro_recall_score)
        self.log(phase + ' Macro F1-score', macro_f1_score)

        class_ids = [(i+1) for i in range(len(f1_per_class))]
        macro_precision_table_data = [[class_id, precision_score] for (class_id, precision_score) in zip(class_ids, precision_per_class)]
        macro_precision_table = wandb.Table(data = macro_precision_table_data, columns=["class", "Precision score"])
        wandb.log({phase + ' Precision score per class': wandb.plot.bar(macro_precision_table, "class", "Precision score", title = phase + " Precision score per class")})

        macro_recall_table_data = [[class_id, recall_score] for (class_id, recall_score) in zip(class_ids, recall_per_class)]
        macro_recall_table = wandb.Table(data = macro_recall_table_data, columns=["class", "Recall score"])
        wandb.log({phase + ' Recall score per class': wandb.plot.bar(macro_recall_table, "class", "Recall score", title = phase + " Recall score per class")})

        macro_f1_table_data = [[class_id, f1_score] for (class_id, f1_score) in zip(class_ids, f1_per_class)]
        macro_f1_table = wandb.Table(data = macro_f1_table_data, columns=["class", "F1-score"])
        wandb.log({phase + ' F1-score per class': wandb.plot.bar(macro_f1_table, "class", "F1-score", title = phase + " F1-score per class")})

        return macro_f1_score

    def tune_single_threshold_bootstrap(self, output, labels):
        num_instances = len(output)
        if num_instances < 1:
            self.best_threshold = 0.5
            return
        num_samples = int(0.1 * num_instances)
        if num_samples < 1:
            num_samples = 1
        bootstrap_scores = []
        for i in range(bootstrap_samples):
            temp_scores = []
            bootstrap_indices = np.random.choice(num_instances, num_samples, replace=True)
            bootstrap_output = output[bootstrap_indices]
            bootstrap_labels = labels[bootstrap_indices]
            for threshold in thresholds:
                f1_metric_macro = MultilabelF1Score(average = 'macro', num_labels = self.num_labels, threshold = threshold).to(self.device)
                f1_metric_micro = MultilabelF1Score(average = 'micro', num_labels = self.num_labels, threshold = threshold).to(self.device)
                macro_f1_score = f1_metric_macro(bootstrap_output, bootstrap_labels).item()
                micro_f1_score = f1_metric_micro(bootstrap_output, bootstrap_labels).item()
                if micro_f1_score + macro_f1_score > 0:
                    harmonic_mean = (2 * micro_f1_score * macro_f1_score)/(micro_f1_score + macro_f1_score)
                else: 
                    harmonic_mean = 0
                temp_scores.append(harmonic_mean)
            bootstrap_scores.append(temp_scores)
        bootstrap_scores = np.mean(bootstrap_scores, axis=0)
        best_threshold = thresholds[np.argmax(bootstrap_scores)]
        self.best_threshold = best_threshold

    def tune_threshold_per_level_bootstrap(self, output, labels):
        prev_levels_num_labels = 0
        for i in range(len(self.level_num_labels)):
            level_output = output[:, prev_levels_num_labels: prev_levels_num_labels + self.level_num_labels[i]].cpu()
            level_labels = labels[:, prev_levels_num_labels: prev_levels_num_labels + self.level_num_labels[i]].cpu()
            prev_levels_num_labels += self.level_num_labels[i]

            num_instances = len(level_output)
            if num_instances < 1:
                self.best_threshold_per_level[i] = 0.5
                continue
            num_samples = int(0.1 * num_instances)
            if num_samples < 1:
                num_samples = 1
            bootstrap_scores = []
            for j in range(bootstrap_samples):
                temp_scores = []
                bootstrap_indices = np.random.choice(num_instances, num_samples, replace=True)
                bootstrap_output = level_output[bootstrap_indices]
                bootstrap_labels = level_labels[bootstrap_indices]
                for threshold in thresholds[i]:
                    if self.level_num_labels[i] > 1:
                        f1_metric_macro = MultilabelF1Score(average = 'macro', num_labels = self.level_num_labels[i], threshold = threshold).to('cpu')
                        f1_metric_micro = MultilabelF1Score(average = 'micro', num_labels = self.level_num_labels[i], threshold = threshold).to('cpu')
                        macro_f1_score = f1_metric_macro(bootstrap_output, bootstrap_labels).item()
                        micro_f1_score = f1_metric_micro(bootstrap_output, bootstrap_labels).item()
                        if micro_f1_score + macro_f1_score > 0:
                            harmonic_mean = (2 * micro_f1_score * macro_f1_score)/(micro_f1_score + macro_f1_score)
                        else: 
                            harmonic_mean = 0
                    else:
                        f1_metric = BinaryF1Score(threshold = threshold).to('cpu')
                        harmonic_mean = f1_metric(bootstrap_output, bootstrap_labels).item()
                    temp_scores.append(harmonic_mean)
                bootstrap_scores.append(temp_scores)
            bootstrap_scores = np.mean(bootstrap_scores, axis=0)
            best_threshold = thresholds[i][np.argmax(bootstrap_scores)]
            self.best_threshold_per_level[i] = best_threshold
    
    def tune_threshold_per_label_bootstrap(self, output, labels):
        for i in range(sum(self.level_num_labels)):
            label_output = output[:, i].cpu()
            label_labels = labels[:, i].cpu()

            num_instances = len(label_output)
            if num_instances < 1:
                self.best_threshold_per_label[i] = 0.5
                continue
            num_samples = int(0.1 * num_instances)
            if num_samples < 1:
                num_samples = 1
            bootstrap_scores = []
            for j in range(bootstrap_samples):
                temp_scores = []
                bootstrap_indices = np.random.choice(num_instances, num_samples, replace=True)
                bootstrap_output = label_output[bootstrap_indices]
                bootstrap_labels = label_labels[bootstrap_indices]
                for threshold in thresholds[i]:
                    f1_metric = BinaryF1Score(threshold = threshold).to('cpu')
                    f1_score = f1_metric(bootstrap_output, bootstrap_labels).item()
                    temp_scores.append(f1_score)
                bootstrap_scores.append(temp_scores)
            bootstrap_scores = np.mean(bootstrap_scores, axis=0)
            best_threshold = thresholds[i][np.argmax(bootstrap_scores)]
            self.best_threshold_per_label[i] = best_threshold

    def validation_step(self, batch, batch_idx):
        label_logits = self.obtain_output(batch, training_data=False)
        labels = batch['labels']

        self.val_epoch_outputs.append(label_logits)
        self.val_epoch_labels.append(labels)

        label_logits_input_to_loss = torch.special.logit(label_logits, eps=1e-6)
        loss = self.loss_fn(label_logits_input_to_loss, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        output = self.obtain_output(batch, training_data=False)
        labels = batch['labels']

        self.test_epoch_outputs.append(output)
        self.test_epoch_labels.append(labels)

    def on_test_epoch_end(self):
        test_epoch_outputs = torch.cat(self.test_epoch_outputs, dim = 0)
        test_epoch_labels = torch.cat(self.test_epoch_labels, dim = 0)

        for i in range(len(threshold_tuning_methods)):
            global threshold_tuning_method
            threshold_tuning_method = threshold_tuning_methods[i]
            self.log_metrics(test_epoch_outputs, test_epoch_labels, threshold_tuning_method + '_Test')

        self.test_epoch_outputs.clear()
        self.test_epoch_labels.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        if use_lr_scheduler:
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

def get_data(dataset_name):
    if dataset_name == 'WOS':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, 'WebOfScience'),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, 'WebOfScience'),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, 'WebOfScience')})
    elif dataset_name == 'NYT':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})
    elif dataset_name == 'RCV1':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})
        
    elif dataset_name == 'HCRD':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})
    elif dataset_name == 'CREST':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})
    elif dataset_name == 'MESO':
        dataset = datasets.load_dataset('json',
                                data_files={'train': data_path + '/{}/{}_train.json'.format(dataset_name, dataset_name),
                                            'dev': data_path + '/{}/{}_dev.json'.format(dataset_name, dataset_name),
                                            'test': data_path + '/{}/{}_test.json'.format(dataset_name, dataset_name)})

    label_dict = torch.load(data_path + '/' + dataset_name + '/value_dict.pt')
    num_labels = len(label_dict)
    depth2label = torch.load(data_path + '/' + dataset_name + '/depth2label.pt')
    path_list = torch.load(data_path + '/' + dataset_name + '/path_list.pt')

    if not num_train_samples == 'all':
        dataset['train'] = dataset['train'].select(range(num_train_samples))
    if not num_dev_samples == 'all':
        dataset['dev'] = dataset['dev'].select(range(num_dev_samples))
    if not num_test_samples == 'all':
        dataset['test'] = dataset['test'].select(range(num_test_samples))

    if low_resource:
        num_training_samples = len(dataset['train'])
        random_sample = np.random.choice(num_training_samples, int(num_training_samples * 0.1), replace=False)
        dataset['train'] = dataset['train'].select(random_sample)

    level_num_labels = []
    for level_labels in depth2label.values():
        level_num_labels.append(len(level_labels))
    return dataset, num_labels, level_num_labels, label_dict, depth2label, path_list

def get_dataset(model, dataset, level_num_labels, depth2label, label_dict, path_list):
    if initialise_embeddings == 'hard':
        train_dataset, val_dataset, test_dataset = model.tokenize_hard(dataset, level_num_labels, depth2label, label_dict) 
    elif initialise_embeddings == 'soft':
        train_dataset, val_dataset, test_dataset = model.tokenize_soft(dataset, level_num_labels, depth2label, label_dict, path_list, graph_type, graph_layers, add_prompts_after_hier_prompt) 
    elif initialise_embeddings == 'flat':
        train_dataset, val_dataset, test_dataset = model.tokenize_flat(dataset, level_num_labels, depth2label, label_dict, path_list, graph_type, graph_layers, add_prompts_after_hier_prompt) 

    train_dataset = Dataset.from_dict(train_dataset)
    val_dataset = Dataset.from_dict(val_dataset)
    test_dataset = Dataset.from_dict(test_dataset)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'position_ids', 'token_type_ids', 'positions', 'labels', 'text_length'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'position_ids', 'token_type_ids', 'positions', 'labels', 'text_length'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'position_ids', 'token_type_ids', 'positions', 'labels', 'text_length'])
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
        for dataset_name in dataset_names:
            if not hyperparam_tune:
                random_seeds = [33, 44, 55]
                if model_type == 'electra':
                    if dataset_name == 'WOS':
                        learning_rate = [8e-6]
                    elif dataset_name == 'RCV1':
                        learning_rate = [1.5e-5]
                    elif dataset_name == 'NYT':
                        learning_rate = [2e-5]
            dataset, num_labels, level_num_labels, label_dict, depth2label, path_list = get_data(dataset_name)
            for random_seed in random_seeds:
                pl.seed_everything(random_seed, workers=True)
                base_model = PromptModel.from_pretrained(model_name, num_labels = num_labels, dlm_loss = dlm_loss, generator_name = generator_name, text_mask = text_mask, template_at_end = template_at_end, class_bias = True)
                train_dataset, val_dataset, test_dataset = get_dataset(base_model, dataset, level_num_labels, depth2label, label_dict, path_list)
                torch.save(base_model, 'base_model.pt')
                del base_model
                for class_bias in class_biases:
                    for dlm_loss_ratio in dlm_loss_ratios:
                        sweep_config = {
                            'name': sweep_name,
                            'method': 'grid',
                        }
                        metric = {
                            'name': 'val_loss',
                            'goal': 'minimize'
                        }
                        sweep_config['metric'] = metric
                        parameters_dict = {
                            'lr': {
                                'values': learning_rate
                            },
                            'batch_size': {
                                'values': batch_size
                            },
                        }
                        sweep_config['parameters'] = parameters_dict
                        sweep_id = wandb.sweep(sweep_config, project="Masters")
                        wandb.agent(sweep_id, Train)
