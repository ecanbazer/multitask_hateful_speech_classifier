import torch
from transformers import AdamW
from torch.utils.data import DataLoader

import copy
from tqdm import trange

import logging
logger = logging.getLogger('sequence_tagger_bert')



class ModelTrainerBert:
    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler,
                 train_dataset, 
                 val_dataset, 
                 update_scheduler='es', # ee(every_epoch) or every_step(es)
                 keep_best_model=True,
                 restore_bm_on_lr_change=False,
                 max_grad_norm=1.0,
                 smallest_lr=0.,
                 validation_metrics=None,
                 decision_metric=None,
                 loader_args={'num_workers' : 1},
                 batch_size=32):
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
            
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        
        self._update_scheduler = update_scheduler
        self._keep_best_model = keep_best_model
        self._restore_bm_on_lr_change = restore_bm_on_lr_change
        self._max_grad_norm = max_grad_norm
        self._smallest_lr = smallest_lr
        self._validation_metrics = validation_metrics
        self._decision_metric = decision_metric
        if self._decision_metric is None:
            self._decision_metric = lambda metrics: metrics[2]
            
        self._loader_args = loader_args
        self._batch_size = batch_size
    
    def _make_tensors(self, dataset_row): 
        tokens, tag_labels, class_labels = tuple(zip(*dataset_row))
        return self._model.generate_tensors_for_training(tokens, tag_labels, class_labels)
    
    def train(self, epochs, task1coef = 1, task2coef = 1):
        best_model = {}
        best_dec_metric = float(0)
        
        get_lr = lambda: self._optimizer.param_groups[0]['lr']
        
        train_dataloader = DataLoader(self._train_dataset, 
                                      batch_size=self._batch_size, 
                                      shuffle=True,
                                      collate_fn=self._make_tensors)
        training_losses1 = []
        training_losses2 = []

        val_losses1 = []
        val_losses2 = []

        iterator = trange(epochs, desc='Epoch')
        for epoch in iterator:
            self._model._bert_model.train()

            cum_loss = 0.
            for nb, tensors in enumerate(train_dataloader):
                loss1, loss2 = self._model.batch_loss_tensors(*tensors)
                training_losses1.append(loss1.mean().item())
                training_losses2.append(loss2.mean().item())
                loss = (task1coef * loss1) + (task2coef * loss2)
                cum_loss += loss.item()
                
                self._model._bert_model.zero_grad()
                loss.backward()
                if self._max_grad_norm > 0.:
                    torch.nn.utils.clip_grad_norm_(parameters=self._model._bert_model.parameters(), 
                                                   max_norm=self._max_grad_norm)
                    
                self._optimizer.step()
        
                if self._update_scheduler == 'es':
                    self._lr_scheduler.step()
            
            #printing parameters
#            for name, param in self._model._bert_model.named_parameters():
#                if param.requires_grad:
#                    print("name: ",name, "parameter grad value: ", param.grad.data.sum())      
                        
            prev_lr = get_lr()
            logger.info(f'Current learning rate: {prev_lr}')
            
            cum_loss /= (nb + 1)
            logger.info(f'Train loss: {cum_loss}')

            dec_metric = 0.
            if self._val_dataset is not None:
                _, val_pred_tags, ___, val_metrics, val_loss1, val_loss2 = self._model.predict(self._val_dataset, evaluate=True, 
                                                         metrics=self._validation_metrics)
                val_loss = val_metrics[0]
                val_losses1.append(val_loss1.mean().item())
                val_losses2.append(val_loss2.mean().item())
                logger.info(f'Validation loss: {val_loss}')
                logger.info(f'Validation metrics: {val_metrics[1:]}')
                
                dec_metric = self._decision_metric(val_metrics)

                print()
                print('DEC METRIC', dec_metric)
                print()
                print('BEST_DEC_METRIC', best_dec_metric)
                print()


                if self._keep_best_model and (dec_metric > best_dec_metric):
                    best_model = copy.deepcopy(self._model._bert_model.state_dict())
                    best_dec_metric = dec_metric
            
            if self._update_scheduler == 'ee':
                self._lr_scheduler.step(dec_metric)
                
            if self._restore_bm_on_lr_change and get_lr() < prev_lr:
                if get_lr() < self._smallest_lr: 
                    iterator.close()
                    break

                prev_lr = get_lr()
                logger.info(f'Reduced learning rate to: {prev_lr}')
                    
                logger.info('Restoring best model...')
                self._model._bert_model.load_state_dict(best_model)




        if best_model:
            self._model._bert_model.load_state_dict(best_model)


        torch.cuda.empty_cache()

