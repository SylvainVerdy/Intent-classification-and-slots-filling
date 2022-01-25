import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, classes, classes_slots) -> None:
        super().__init__()

        self.classes = classes
        self.num_slot_labels = classes_slots
        self.dim = 768
        self.pe = PositionalEncoding(self.dim, max_len = 5000)

        # encoder  layers
        self.emb = torch.nn.Embedding(num_embeddings=30555, embedding_dim=self.dim)

        enc_layer = torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, dim_feedforward=2048, dropout=0.2) 
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers = 2)
        self.intent_classifier = IntentClassifier(self.dim, self.classes, 0.1)
        self.slot_classifier = SlotClassifier(self.dim, self.num_slot_labels, 0.1)


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        src = self.emb(input_ids).permute(1,0,2)
        x = self.pe(src)
        x = self.encoder(x)
        pooled_output = x[0,:,:]
        sequence_output = x[:, :, :].permute(1,0,2)
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.classes == 1:
                intent_loss_fct = nn.MSELoss()
                
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids)
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.classes), intent_label_ids)
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:

            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Only keep active parts of the loss
        
            slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            total_loss += 0.5 * slot_loss

        outputs = ((intent_logits, slot_logits),)  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
