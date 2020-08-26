import torch
from torch import nn
from torch.nn import Module, Linear, LayerNorm
from transformers import BertPreTrainedModel, LongformerModel, RobertaModel, RobertaConfig
from transformers.modeling_bert import ACT2FN


class FullyConnectedLayer(Module):
    # TODO: many layers
    def __init__(self, config, input_dim, output_dim):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        return temp


class CoreferenceResolutionModel(BertPreTrainedModel):
    def __init__(self, config, args, antecedent_loss, max_span_length, seperate_mention_loss):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.antecedent_loss = antecedent_loss  # can be either allowed loss or bce
        self.max_span_length = max_span_length
        self.seperate_mention_loss = seperate_mention_loss
        self.args = args

        if args.model_type == "longformer":
            self.longformer = LongformerModel(config)
        elif args.model_type == "roberta":
            self.roberta = RobertaModel(config)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size)
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size)
        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size)
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, config.hidden_size)

        self.entity_mention_start_classifier = nn.Linear(config.hidden_size, 1)  # In paper w_s
        self.entity_mention_end_classifier = nn.Linear(config.hidden_size, 1)  # w_e
        self.entity_mention_joint_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # M

        self.antecedent_start_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # S
        self.antecedent_end_classifier = nn.Linear(config.hidden_size, config.hidden_size)  # E

        self.init_weights()

    def _compute_joint_entity_mention_loss(self, start_entity_mention_labels, end_entity_mention_labels, mention_logits,
                                           attention_mask=None):
        """
        :param start_entity_mention_labels: [batch_size, num_mentions]
        :param end_entity_mention_labels: [batch_size, num_mentions]
        :param mention_logits: [batch_size, seq_length, seq_length]
        :return:
        """
        device = start_entity_mention_labels.device
        batch_size, seq_length, _ = mention_logits.size()
        num_mentions = start_entity_mention_labels.size(-1)

        # We now take the index tensors and turn them into sparse tensors
        labels = torch.zeros(size=(batch_size, seq_length, seq_length), device=device)
        batch_temp = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, num_mentions)
        labels[
            batch_temp, start_entity_mention_labels, end_entity_mention_labels] = 1.0  # [batch_size, seq_length, seq_length]
        labels[:, 0, 0] = 0.0  # Remove the padded mentions

        weights = (attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2))
        mention_mask = self._get_mention_mask(weights)
        weights = weights * mention_mask

        loss = self._compute_pos_neg_loss(weights, labels, mention_logits)
        return loss

    def _calc_boundary_loss(self, boundary_entity_mention_labels, boundary_mention_logits, attention_mask):
        device = boundary_entity_mention_labels.device
        batch_size, seq_length = boundary_mention_logits.size()
        num_mentions = boundary_entity_mention_labels.size(-1)
        # We now take the index tensors and turn them into sparse tensors
        boundary_labels = torch.zeros(size=(batch_size, seq_length), device=device)
        batch_temp = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, num_mentions)
        boundary_labels[
            batch_temp, boundary_entity_mention_labels] = 1.0  # [batch_size, seq_length]
        boundary_labels[:, 0] = 0.0  # Remove the padded mentions

        boundary_weights = attention_mask
        loss = self._compute_pos_neg_loss(boundary_weights, boundary_labels, boundary_mention_logits)
        return loss

    def _compute_seperate_entity_mention_loss(self, start_entity_mention_labels, end_entity_mention_labels,
                                              start_mention_logits, end_mention_logits, joint_mention_logits,
                                              attention_mask=None):
        joint_loss = self._compute_joint_entity_mention_loss(start_entity_mention_labels, end_entity_mention_labels,
                                                             joint_mention_logits,
                                                             attention_mask)
        start_loss = self._calc_boundary_loss(start_entity_mention_labels, start_mention_logits, attention_mask)
        end_loss = self._calc_boundary_loss(end_entity_mention_labels, end_mention_logits, attention_mask)
        loss = (joint_loss + start_loss + end_loss) / 3
        return loss

    def _prepare_antecedent_matrix(self, antecedent_labels):
        """
        :param antecedent_labels: [batch_size, seq_length, cluster_size]
        :return: [batch_size, seq_length, seq_length]
        """
        device = antecedent_labels.device
        batch_size, seq_length, cluster_size = antecedent_labels.size()

        # We now prepare a tensor with the gold antecedents for each span
        labels = torch.zeros(size=(batch_size, seq_length, seq_length), device=device)
        batch_temp = torch.arange(batch_size, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, seq_length,
                                                                                                cluster_size)
        seq_length_temp = torch.arange(seq_length, device=device).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1,
                                                                                                    cluster_size)

        labels[batch_temp, seq_length_temp, antecedent_labels] = 1.0
        labels[:, :, -1] = 0.0  # Fix all pad-antecedents

        return labels

    def _compute_pos_neg_loss(self, weights, labels, logits):
        pos_weights = weights * labels
        per_example_pos_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        per_example_pos_loss = per_example_pos_loss_fct(logits, labels)
        pos_loss = (per_example_pos_loss * pos_weights).sum() / (pos_weights.sum() + 1e-8)

        neg_weights = weights * (1 - labels)
        per_example_neg_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        per_example_neg_loss = per_example_neg_loss_fct(logits, labels)
        neg_loss = (per_example_neg_loss * neg_weights).sum() / (neg_weights.sum() + 1e-8)

        loss = neg_loss + pos_loss
        return loss

    def _compute_antecedent_loss(self, antecedent_labels, antecedent_logits, attention_mask=None):
        """
        :param antecedent_labels: [batch_size, seq_length, cluster_size]
        :param antecedent_logits: [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        """
        seq_length = antecedent_logits.size(-1)
        labels = self._prepare_antecedent_matrix(antecedent_labels)  # [batch_size, seq_length, seq_length]
        gold_antecedent_logits = antecedent_logits + ((1 - labels) * -1e8)

        if self.antecedent_loss == "allowed":
            only_positive_labels = labels.clone()
            only_positive_labels[:, :, 0] = 0
            num_positive_examples = torch.sum(only_positive_labels)
            num_negative_examples = torch.sum(labels[:, :, 0]) - torch.sum(1 - attention_mask)
            pos_weight = num_negative_examples / num_positive_examples

            gold_log_sum_exp = torch.logsumexp(gold_antecedent_logits, dim=-1)  # [batch_size, seq_length]
            all_log_sum_exp = torch.logsumexp(antecedent_logits, dim=-1)  # [batch_size, seq_length]

            gold_log_probs = gold_log_sum_exp - all_log_sum_exp
            losses = -gold_log_probs

            loss_weights = (torch.sum(only_positive_labels, dim=-1) * (pos_weight - 1)) + 1
            losses = losses * loss_weights

            attention_mask_to_add = torch.zeros_like(attention_mask)
            attention_mask_to_add[:, 0] = -1
            attention_mask = attention_mask + attention_mask_to_add
            sum_losses = torch.sum(losses * attention_mask)
            num_examples = torch.sum(attention_mask)
            loss = sum_losses / (num_examples + 1e-8)
        else:  # == bce
            weights = torch.ones_like(labels).tril()
            weights[:, 0, 0] = 1
            attention_mask = attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2)
            weights = weights * attention_mask

            # Compute pos-neg loss for all non-null antecedents
            non_null_weights = weights.clone()
            non_null_weights[:, :, 0] = 0
            non_null_loss = self._compute_pos_neg_loss(non_null_weights, labels, antecedent_logits)

            # Compute pos-neg loss for all null antecedents
            null_weights = weights.clone()
            null_weights[:, :, 1:] = 0
            null_loss = self._compute_pos_neg_loss(null_weights, labels, antecedent_logits)

            loss = null_loss + non_null_loss

        return loss

    def mask_antecedent_logits(self, antecedent_logits):
        antecedents_mask = torch.ones_like(antecedent_logits).triu(diagonal=1) * (
            -1e8)  # [batch_size, seq_length, seq_length]
        antecedents_mask[:, 0, 0] = 0
        antecedent_logits = antecedent_logits + antecedents_mask  # [batch_size, seq_length, seq_length]
        return antecedent_logits

    def _get_mention_mask(self, mention_logits_or_weights):
        """
        Returns a tensor of size [batch_size, seq_length, seq_length] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits_or_weights: Either the span mention logits or weights, size [batch_size, seq_length, seq_length]
        """
        mention_mask = torch.ones_like(mention_logits_or_weights)
        mention_mask = mention_mask.triu(diagonal=0)
        mention_mask = mention_mask.tril(diagonal=self.max_span_length - 1)
        return mention_mask

    def _get_encoder(self):
        if self.args.model_type == "longformer":
            return self.longformer
        elif self.args.model_type == "roberta":
            return self.roberta

        raise ValueError("Unsupported model type")

    def forward(self, input_ids, attention_mask=None, start_entity_mention_labels=None, end_entity_mention_labels=None,
                start_antecedent_labels=None, end_antecedent_labels=None, return_all_outputs=False):
        encoder = self._get_encoder()
        outputs = encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output)
        end_mention_reps = self.end_mention_mlp(sequence_output)
        start_coref_reps = self.start_coref_mlp(sequence_output)
        end_coref_reps = self.end_coref_mlp(sequence_output)

        # Entity mention scores
        start_mention_logits = self.entity_mention_start_classifier(start_mention_reps).squeeze(
            -1)  # [batch_size, seq_length]
        end_mention_logits = self.entity_mention_end_classifier(end_mention_reps).squeeze(
            -1)  # [batch_size, seq_length]

        temp = self.entity_mention_joint_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        mention_mask = self._get_mention_mask(mention_logits)
        mention_mask = (1 - mention_mask) * -1e8
        mention_logits = mention_logits + mention_mask

        # Antecedent scores
        temp = self.antecedent_start_classifier(start_coref_reps)  # [batch_size, seq_length, dim]
        start_coref_logits = torch.matmul(temp,
                                          start_coref_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]
        start_coref_logits = self.mask_antecedent_logits(start_coref_logits)
        temp = self.antecedent_end_classifier(end_coref_reps)  # [batch_size, seq_length, dim]
        end_coref_logits = torch.matmul(temp, end_coref_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]
        end_coref_logits = self.mask_antecedent_logits(end_coref_logits)

        outputs = outputs[2:]
        if return_all_outputs:
            outputs = (mention_logits, start_coref_logits, end_coref_logits) + outputs

        if start_entity_mention_labels is not None and end_entity_mention_labels is not None \
                and start_antecedent_labels is not None and end_antecedent_labels is not None:
            if self.seperate_mention_loss:
                entity_mention_loss = self._compute_seperate_entity_mention_loss(
                    start_entity_mention_labels=start_entity_mention_labels,
                    end_entity_mention_labels=end_entity_mention_labels,
                    start_mention_logits=start_mention_logits,
                    end_mention_logits=end_mention_logits,
                    joint_mention_logits=joint_mention_logits,
                    attention_mask=attention_mask
                )
            else:
                entity_mention_loss = self._compute_joint_entity_mention_loss(
                    start_entity_mention_labels=start_entity_mention_labels,
                    end_entity_mention_labels=end_entity_mention_labels,
                    mention_logits=mention_logits,
                    attention_mask=attention_mask)
            start_coref_loss = self._compute_antecedent_loss(antecedent_labels=start_antecedent_labels,
                                                             antecedent_logits=start_coref_logits,
                                                             attention_mask=attention_mask)
            end_coref_loss = self._compute_antecedent_loss(antecedent_labels=end_antecedent_labels,
                                                           antecedent_logits=end_coref_logits,
                                                           attention_mask=attention_mask)
            loss = entity_mention_loss + start_coref_loss + end_coref_loss
            outputs = (loss,) + outputs + (entity_mention_loss, start_coref_loss, end_coref_loss)

        return outputs
