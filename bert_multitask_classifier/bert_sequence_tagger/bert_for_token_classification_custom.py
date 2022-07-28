from pytorch_transformers import BertForTokenClassification
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
softmax = nn.Softmax(dim=1)

class BertForTokenClassificationCustom(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_tag = config.num_labels
        self.num_labels_class = 2
#        self.dropout1 = nn.Dropout(classifier_dropout)
#        self.dropout2 = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tag_labels=None,
                class_labels = None, position_ids=None, head_mask=None, loss_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        logits1 = self.classifier(sequence_output)
        logits2 = self.classifier2(pooled_output)
#        logits2 = softmax(logits2)

        outputs1 = (logits1,) + outputs[2:]  # add hidden states and attention if they are here
        if tag_labels is not None:
            loss_fct1 = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = (attention_mask.view(-1) == 1) 
                if loss_mask is not None:
                    active_loss &= loss_mask.view(-1)
                    
                active_logits = logits1.view(-1, self.num_labels)[active_loss]
                active_labels = tag_labels.view(-1)[active_loss]
                loss1 = loss_fct1(active_logits, active_labels)
            else:
                loss1 = loss_fct1(logits1.view(-1, self.num_labels), tag_labels.view(-1))
            outputs1 = (loss1,) + outputs1

        # task2 outputs
        outputs2 = (logits2,) + outputs[2:]
        if class_labels is not None:
            loss_fct2 = CrossEntropyLoss()
#            print()
 #           print('logits2: ', logits2.view(-1, 2))
  #          print()
   #         print('class_labels: ', class_labels.view(-1))
            loss2 = loss_fct2(logits2.view(-1,2), class_labels.view(-1))
            outputs2 = (loss2,) + outputs2


        return outputs1, outputs2  # (loss), scores, (hidden_states), (attentions)
    
