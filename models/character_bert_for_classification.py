from transformers.models.bert.modeling_bert import BertForSequenceClassification
from torch import nn

class CharacterBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = None 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        num_chunks=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        print("[forward] input_ids: ", input_ids.size())
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        seq_len = input_ids.size(2)
        token_len = input_ids.size(3)
        
        # (batch_size x num_chunks) x 512 x 50
        batched_input_ids = input_ids.view(-1, seq_len, token_len)
        # (batch_size x num_chunks) x 512 
        batched_attention_mask = attention_mask.view(-1, seq_len)
        batched_token_type_ids = token_type_ids.view(-1, seq_len)

        outputs = self.bert(
            batched_input_ids,
            attention_mask=batched_attention_mask,
            token_type_ids=batched_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print("[forward] outputs: ", outputs.size())

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )