# HDPT model

# Imports
from transformers import ElectraForPreTraining, ElectraForMaskedLM, AutoTokenizer, ElectraPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Global variables
from graph import MyGraphEncoder
from constants import frozen, num_soft_prompts, random_prompt, token_type_setting, add_SEP

class PromptModel(ElectraPreTrainedModel):
    def __init__(self, config, dlm_loss, generator_name, text_mask, template_at_end, class_bias):
        super().__init__(config)
        self.model_name = config._name_or_path
        self.num_labels = config.num_labels
        self.embedding_size = config.embedding_size
        self.max_length = config.max_position_embeddings

        self.electra = ElectraForPreTraining.from_pretrained(self.model_name, config=config)

        if frozen:
            for param in self.electra.parameters():
                    param.requires_grad = False
            
            self.electra.electra.embeddings.requires_grad_(True)

        self.dlm_loss = dlm_loss
        self.text_mask = text_mask
        if self.dlm_loss:
            self.generator = ElectraForMaskedLM.from_pretrained(generator_name)
            for param in self.generator.parameters():
                param.requires_grad = False
        self.template_at_end = template_at_end
        self.class_bias = class_bias
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bias = nn.Parameter(torch.zeros(self.num_labels))

        self.label_word_ids = []

    def MyGraphEmbedding(self, level_num_labels, label_dict, path_list, graph_type, graph_layers):
        label_dict = {i: self.tokenizer.encode(v, add_special_tokens=False) for i, v in label_dict.items()}

        label_emb = []
        # Obtains the nn.Embedding Module that maps token ids to word embeddings from the pre-trained ELECTRA model.
        input_embeds = self.electra.get_input_embeddings()
        # Uses the nn.Embedding module to obtain the word embeddings for each label description and averages the embeddings to obtain 'label_emb'
        for i in range(len(label_dict)):
            label_emb.append(input_embeds.weight.index_select(0, torch.tensor(label_dict[i])).mean(dim=0))

        label_emb = torch.stack(label_emb)
        embedding_size = self.embedding_size
        num_levels = len(level_num_labels)
        prompt_embedding = nn.Embedding(num_levels, embedding_size)
        label_emb = torch.cat([label_emb, prompt_embedding.weight[:, :]], dim = 0)
        graph_encoder = MyGraphEncoder(graph_type = graph_type, layers = graph_layers, path_list = path_list, embedding_size = embedding_size)
        output_embeddings = graph_encoder(label_emb)
        label_embeddings = output_embeddings[:-num_levels]
        prompt_embeddings = output_embeddings[-num_levels:]
        return label_embeddings, prompt_embeddings
    
    def label_embeddings_from_descriptions(self, label_dict):
        # NOTE: we are removing the CLS and SEP token at the start and end of the description.
        label_dict = {i: self.tokenizer.encode(v, add_special_tokens=False) for i, v in label_dict.items()}
        label_emb = []
        # Obtains the nn.Embedding Module that maps token ids to word embeddings from the pre-trained ELECTRA model.
        input_embeds = self.electra.get_input_embeddings()
        # Uses the nn.Embedding module to obtain the word embeddings for each label description and averages the embeddings to obtain 'label_emb'
        for i in range(len(label_dict)):
            label_emb.append(input_embeds.weight.index_select(0, torch.tensor(label_dict[i])).mean(dim=0))
        return label_emb

    def tokenize_hard(self, dataset, level_num_labels, depth2label, label_dict):
        prompts = []
        len_prompt = []
        for i in range(len(level_num_labels)):
            prompts.append(self.tokenizer.tokenize("Level {} label:".format(i + 1)))
            len_prompt.append(len(prompts[i]))

        # Obtain the label words and their ids.
        label_words = []
        label_word_ids_flat = []
        prev_levels_num_labels = 0
        for i in range(len(level_num_labels)):
            level_label_words = []
            level_label_words_ids = []
            for j in range(level_num_labels[i]):
                temp_label_word_token = "[Class{}]".format(prev_levels_num_labels + j)
                self.tokenizer.add_tokens([temp_label_word_token])
                level_label_words.append(temp_label_word_token)
                level_label_words_ids.append(self.tokenizer.convert_tokens_to_ids([temp_label_word_token])[0])
            prev_levels_num_labels += level_num_labels[i]
            label_words.append(level_label_words)
            self.label_word_ids.append(level_label_words_ids)
            label_word_ids_flat += self.label_word_ids[i]
        self.electra.resize_token_embeddings(len(self.tokenizer))
        if self.dlm_loss:
            self.generator.resize_token_embeddings(len(self.tokenizer))

        postfix = []
        for i in range(len(level_num_labels)):
            postfix.extend(prompts[i])
            postfix.extend(label_words[i])
        postfix += ["[SEP]"]

        label_embeddings = self.label_embeddings_from_descriptions(label_dict)
        # Initialise the label words with the average of their label name descriptions.
        for i in range(len(label_word_ids_flat)):
            self.electra.electra.embeddings.word_embeddings.weight.data[label_word_ids_flat[i]] = label_embeddings[i]
            if self.dlm_loss:
                self.generator.electra.embeddings.word_embeddings.weight.data[label_word_ids_flat[i]] = label_embeddings[i]
        
        train_dataset = tokenize_data(dataset['train'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length)
        val_dataset = tokenize_data(dataset['dev'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length)
        test_dataset = tokenize_data(dataset['test'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length)
        return train_dataset, val_dataset, test_dataset
    
    def tokenize_soft(self, dataset, level_num_labels, depth2label, label_dict, path_list, graph_type, graph_layers, add_prompts_after_hier_prompt):
        if add_prompts_after_hier_prompt:
            prompts = []
            prompt_ids = []
            len_prompt = []
            prompts_per_level = num_soft_prompts
            prev_level_num_prompts = 0
            for i in range(len(level_num_labels)):
                level_prompts = []
                for j in range(prompts_per_level):
                    temp_prompt = "[Prompt{}]".format(prev_level_num_prompts+j)
                    self.tokenizer.add_tokens([temp_prompt])
                    level_prompts.append(temp_prompt)
                    if j == 0:
                        prompt_ids.append(self.tokenizer.convert_tokens_to_ids(temp_prompt))
                len_prompt.append(len(level_prompts))
                prompts.append(level_prompts)
                prev_level_num_prompts += prompts_per_level
        else:
            prompts = []
            prompt_ids = []
            len_prompt = []
            for i in range(len(level_num_labels)):
                prompts.append("[Prompt{}]".format(i))
                self.tokenizer.add_tokens([prompts[i]])
                prompt_ids.append(self.tokenizer.convert_tokens_to_ids(prompts[i]))
                len_prompt.append(len(prompts[i].split()))

        # Obtain the label words and their ids.
        label_words = []
        label_word_ids_flat = []
        prev_levels_num_labels = 0
        for i in range(len(level_num_labels)):
            level_label_words = []
            level_label_words_ids = []
            for j in range(level_num_labels[i]):
                temp_label_word_token = "[Class{}]".format(prev_levels_num_labels + j)
                self.tokenizer.add_tokens([temp_label_word_token])
                level_label_words.append(temp_label_word_token)
                level_label_words_ids.append(self.tokenizer.convert_tokens_to_ids([temp_label_word_token])[0])
            prev_levels_num_labels += level_num_labels[i]
            label_words.append(level_label_words)
            self.label_word_ids.append(level_label_words_ids)
            label_word_ids_flat += self.label_word_ids[i]
        self.electra.resize_token_embeddings(len(self.tokenizer))
        if self.dlm_loss:
            self.generator.resize_token_embeddings(len(self.tokenizer))

        postfix = []
        for i in range(len(level_num_labels)):
            if add_prompts_after_hier_prompt:
                postfix.extend(prompts[i])
            else:
                postfix.append(prompts[i])
            postfix.extend(label_words[i])
        postfix += ["[SEP]"]
        
        label_embeddings, prompt_embeddings = self.MyGraphEmbedding(level_num_labels, label_dict, path_list, graph_type, graph_layers)
        label_embeddings = self.label_embeddings_from_descriptions(label_dict)
        # Initialise the label words with the average of their label name descriptions.
        for i in range(len(label_word_ids_flat)):
            self.electra.electra.embeddings.word_embeddings.weight.data[label_word_ids_flat[i]] = label_embeddings[i]
            if self.dlm_loss:
                self.generator.electra.embeddings.word_embeddings.weight.data[label_word_ids_flat[i]] = label_embeddings[i]
        if not random_prompt:
            for i in range(len(prompt_ids)):
                self.electra.electra.embeddings.word_embeddings.weight.data[prompt_ids[i]] = prompt_embeddings[i]
                if self.dlm_loss:
                    self.generator.electra.embeddings.word_embeddings.weight.data[prompt_ids[i]] = prompt_embeddings[i]
        
        if self.template_at_end:
            train_dataset = tokenize_data_template_end(dataset['train'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
            val_dataset = tokenize_data_template_end(dataset['dev'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
            test_dataset = tokenize_data_template_end(dataset['test'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
        else:
            train_dataset = tokenize_data(dataset['train'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
            val_dataset = tokenize_data(dataset['dev'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
            test_dataset = tokenize_data(dataset['test'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
        return train_dataset, val_dataset, test_dataset
    
    def tokenize_flat(self, dataset, level_num_labels, depth2label, label_dict, path_list, graph_type, graph_layers, add_prompts_after_hier_prompt):
        if add_prompts_after_hier_prompt:
            prompts = []
            prompt_ids = []
            len_prompt = []
            prompts_per_level = num_soft_prompts
            prev_level_num_prompts = 0
            for i in range(len(level_num_labels)):
                level_prompts = []
                for j in range(prompts_per_level):
                    temp_prompt = "[Prompt{}]".format(prev_level_num_prompts+j)
                    self.tokenizer.add_tokens([temp_prompt])
                    level_prompts.append(temp_prompt)
                    if j == 0:
                        prompt_ids.append(self.tokenizer.convert_tokens_to_ids(temp_prompt))
                len_prompt.append(len(level_prompts))
                prompts.append(level_prompts)
                prev_level_num_prompts += prompts_per_level
        else:
            prompts = []
            prompt_ids = []
            len_prompt = []
            for i in range(len(level_num_labels)):
                prompts.append("[Prompt{}]".format(i))
                self.tokenizer.add_tokens([prompts[i]])
                prompt_ids.append(self.tokenizer.convert_tokens_to_ids(prompts[i]))
                len_prompt.append(len(prompts[i].split()))

        # Obtain the label words and their ids.
        label_words = []
        label_word_ids_flat = []
        prev_levels_num_labels = 0
        for i in range(len(level_num_labels)):
            level_label_words = []
            level_label_words_ids = []
            for j in range(level_num_labels[i]):
                temp_label_word_token = "[Class{}]".format(prev_levels_num_labels + j)
                self.tokenizer.add_tokens([temp_label_word_token])
                level_label_words.append(temp_label_word_token)
                level_label_words_ids.append(self.tokenizer.convert_tokens_to_ids([temp_label_word_token])[0])
            prev_levels_num_labels += level_num_labels[i]
            label_words.append(level_label_words)
            self.label_word_ids.append(level_label_words_ids)
            label_word_ids_flat += self.label_word_ids[i]
        self.electra.resize_token_embeddings(len(self.tokenizer))
        if self.dlm_loss:
            self.generator.resize_token_embeddings(len(self.tokenizer))

        postfix = []
        for i in range(len(level_num_labels)):
            if add_prompts_after_hier_prompt:
                postfix.extend(prompts[i])
            else:
                postfix.append(prompts[i])
        for i in range(len(level_num_labels)):
            postfix.extend(label_words[i])
        postfix += ["[SEP]"]
        
        label_embeddings, prompt_embeddings = self.MyGraphEmbedding(level_num_labels, label_dict, path_list, graph_type, graph_layers)
        label_embeddings = self.label_embeddings_from_descriptions(label_dict)
        # Initialise the label words with the average of their label name descriptions.
        for i in range(len(label_word_ids_flat)):
            self.electra.electra.embeddings.word_embeddings.weight.data[label_word_ids_flat[i]] = label_embeddings[i]
            if self.dlm_loss:
                self.generator.electra.embeddings.word_embeddings.weight.data[label_word_ids_flat[i]] = label_embeddings[i]
        if not random_prompt:
            for i in range(len(prompt_ids)):
                self.electra.electra.embeddings.word_embeddings.weight.data[prompt_ids[i]] = prompt_embeddings[i]
                if self.dlm_loss:
                    self.generator.electra.embeddings.word_embeddings.weight.data[prompt_ids[i]] = prompt_embeddings[i]
        
        if self.template_at_end:
            train_dataset = tokenize_data_template_end_flat(dataset['train'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
            val_dataset = tokenize_data_template_end_flat(dataset['dev'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
            test_dataset = tokenize_data_template_end_flat(dataset['test'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
        else:
            train_dataset = tokenize_data(dataset['train'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
            val_dataset = tokenize_data(dataset['dev'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
            test_dataset = tokenize_data(dataset['test'], self.tokenizer, postfix, level_num_labels, self.num_labels, depth2label, len_prompt, self.max_length, self.text_mask)
        return train_dataset, val_dataset, test_dataset

    def forward(self, input_ids, attention_mask, position_ids, token_type_ids, label_positions, text_length, training_data):
        if self.dlm_loss and training_data:
            input_ids, is_replaced = self.get_corrupted_input_ids(input_ids, attention_mask, position_ids, token_type_ids, text_length)

        outputs = self.electra(input_ids, attention_mask = attention_mask, position_ids = position_ids, token_type_ids = token_type_ids)
        raw_logits = outputs[0]
        if self.dlm_loss and training_data:
                text_logits = []
                text_is_replaced = []
                for i in range(len(raw_logits)):
                    text_logits.extend(raw_logits[i, :text_length[i]])
                    text_is_replaced.extend(is_replaced[i, :text_length[i]])
                text_logits = torch.Tensor(text_logits)
                text_is_replaced = torch.Tensor(text_is_replaced)

        label_logits = [raw_logits[i, label_positions[i]] for i in range(len(label_positions))]
        label_logits = torch.stack(label_logits)
        if self.class_bias:
            label_logits = label_logits + self.bias
        label_logits = 1 - torch.sigmoid(label_logits)

        if self.dlm_loss and training_data:
            return label_logits, text_logits, text_is_replaced
        else:
            return label_logits
    
    def get_corrupted_input_ids(self, input_ids, attention_mask, position_ids, token_type_ids, text_length):
        replace_prob=0.1
        original_prob=0.1
        mlm_probability = 0.15
        
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability, device=self.device)

        # Assign 0 probability of being corrupted to tokens after the text and special tokens.
        for i in range(len(probability_matrix)):
            probability_matrix[i, text_length[i]:] = 0.0
        special_token_indices = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map.values())
        special_tokens_mask = torch.full(input_ids.shape, False, dtype=torch.bool, device=self.device)
        for sp_id in special_token_indices:
            special_tokens_mask = special_tokens_mask | (input_ids==sp_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        mlm_mask = torch.bernoulli(probability_matrix).bool()
        labels[~mlm_mask] = -100

        mask_prob = 1 - replace_prob - original_prob
        mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=self.device)).bool() & mlm_mask
        input_ids[mask_token_mask] = self.tokenizer.mask_token_id

        if int(replace_prob)!=0:
            rep_prob = replace_prob/(replace_prob + original_prob)
            replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=self.device)).bool() & mlm_mask & ~mask_token_mask
            random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long, device=self.device)
            input_ids[replace_token_mask] = random_words[replace_token_mask]

        gen_output = self.generator(input_ids, attention_mask=attention_mask, position_ids = position_ids, token_type_ids=token_type_ids)
        gen_logits = gen_output.logits

        mlm_gen_logits = gen_logits[mlm_mask, :]

        with torch.no_grad():
            pred_toks = torch.multinomial(F.softmax(mlm_gen_logits, dim=-1), 1).squeeze()
            generated = input_ids.clone()
            generated[mlm_mask] = pred_toks
            is_replaced = mlm_mask.clone()
            is_replaced[mlm_mask] = (pred_toks != labels[mlm_mask])

        return generated, is_replaced
    
def tokenize_data(dataset, tokenizer, postfix, level_num_labels, num_labels, depth2label, len_prompt, max_length = 512, text_mask = False):
    data_dict = {"input_ids" : [], "attention_mask" : [], "position_ids": [], "token_type_ids": [], "positions" : [], "labels" : [], "text_length" : []}
    for text, label in zip(dataset['token'], dataset['label']):
        # Add input_ids

        text_tokens = ["[CLS]"] + tokenizer.tokenize(text)

        # Check if the text is too long.
        len_text = len(text_tokens)
        len_postfix = len(postfix)
        num_levels = len(level_num_labels)

        # The max length the text tokens can take up, -1 is for the SEP token at end of postfix.
        max_len_text = max_length - num_levels - sum(len_prompt) - 1

        overflow_len = len_text - max_len_text
        if overflow_len > 0:
            text_tokens = text_tokens[:len_text - overflow_len]
        len_trunc_text = len(text_tokens)

        data_dict["text_length"].append(len_trunc_text)
        
        # Adds prompts to end of text.
        text_tokens += postfix
        text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        total_token_len = max_len_text + len_postfix

        # Pad input_ids
        len_tokens = len(text_token_ids)
        if len_tokens < total_token_len:
            text_token_ids = text_token_ids + [0] * (total_token_len - len_tokens)
        data_dict["input_ids"].append(text_token_ids)

        # Add attention_mask
        if text_mask:
            # text attention mask
            attention_mask = [[1] * len_trunc_text + [0] * (total_token_len - len_trunc_text)] * len_trunc_text
            # postfix attention mask
            attention_mask += [[1] * len_tokens + [0] * (total_token_len - len_tokens)] * len_postfix
            # padding attention mask
            attention_mask += [[0] * total_token_len] * (total_token_len - len_tokens)
        else:
            attention_mask = [1] * len_tokens + [0] * (total_token_len - len_tokens)
        data_dict["attention_mask"].append(attention_mask)

        # Add position_ids
        source_position_ids = [i for i in range(len_trunc_text)]
        postfix_position_ids = []
        for k in range(len(level_num_labels)):
            if k == 0:
                postfix_position_ids += [len_trunc_text + i for i in range(len_prompt[k])]
                postfix_position_ids += [len_trunc_text + len_prompt[k]] * level_num_labels[k]
            else:
                postfix_position_ids += [max(postfix_position_ids) + 1 + i for i in range(len_prompt[k])]
                position_id = max(postfix_position_ids) + 1
                postfix_position_ids += [position_id] * level_num_labels[k]

        # Last one for [SEP] token at the end.
        postfix_position_ids.append(max(postfix_position_ids) + 1)
        padding_position_ids = [0] * (total_token_len - len_tokens)
        position_ids = source_position_ids + postfix_position_ids + padding_position_ids
        data_dict["position_ids"].append(position_ids)

        token_type_ids = [0] * total_token_len
        data_dict["token_type_ids"].append(token_type_ids)

        # Add positions of labels
        positions = []
        for k in range(len(level_num_labels)):
            if k == 0:
                # First level positions
                positions += [len_trunc_text + len_prompt[k] + j for j in range(level_num_labels[k])]
            else:
                # Add final positions of previous level (+1) as starting point for next level
                positions += [max(positions) + 1 + len_prompt[k] + j for j in range(level_num_labels[k])]
        data_dict["positions"].append(positions)

        # Add one-hot-encoded labels
        one_hot_labels = np.zeros(num_labels)
        one_hot_labels[label] = 1
        data_dict['labels'].append(torch.from_numpy(np.float32(one_hot_labels)))
    return data_dict

# This tokenizes the data by always putting the template in the same position, after the padding.
def tokenize_data_template_end(dataset, tokenizer, postfix, level_num_labels, num_labels, depth2label, len_prompt, max_length = 512, text_mask = False):
    data_dict = {"input_ids" : [], "attention_mask" : [], "position_ids": [], "token_type_ids": [], "positions" : [], "labels" : [], "text_length" : []}
    for text, label in zip(dataset['token'], dataset['label']):
        # Add input_ids

        text_tokens = ["[CLS]"] + tokenizer.tokenize(text)

        if add_SEP:
            text_tokens += ["[SEP]"]

        # Check if the text is too long.
        len_text = len(text_tokens)
        len_postfix = len(postfix)
        num_levels = len(level_num_labels)

        # The max length the text tokens can take up, -1 is for the SEP token at end of postfix.
        max_len_text = max_length - num_levels - sum(len_prompt) - 1

        overflow_len = len_text - max_len_text
        if overflow_len > 0:
            text_tokens = text_tokens[:len_text - overflow_len]
        len_trunc_text = len(text_tokens)

        data_dict["text_length"].append(len_trunc_text)

        # Add padding
        len_padding = max_len_text - len_trunc_text
        if len_padding > 0:
            text_tokens += ["[PAD]"] * len_padding

        text_tokens += postfix
        text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        total_token_len = max_len_text + len_postfix
        len_tokens = max_len_text - len_padding
        data_dict["input_ids"].append(text_token_ids)

        # Add attention_mask
        if text_mask:
            # text attention mask
            attention_mask = [[1] * len_trunc_text + [0] * (total_token_len - len_trunc_text)] * len_trunc_text
            # postfix attention mask
            attention_mask += [[1] * len_tokens + [0] * (total_token_len - len_tokens)] * len_postfix
            # padding attention mask
            attention_mask += [[0] * total_token_len] * (total_token_len - len_tokens)
        else:
            attention_mask = [1] * len_trunc_text + [0] * (len_padding) + [1] * len_postfix
        data_dict["attention_mask"].append(attention_mask)

        # Add position_ids
        source_position_ids = [i for i in range(len_trunc_text)]
        postfix_position_ids = []
        for k in range(len(level_num_labels)):
            if k == 0:
                postfix_position_ids += [len_trunc_text + len_padding + i for i in range(len_prompt[k])]
                postfix_position_ids += [len_trunc_text + len_padding + len_prompt[k]] * level_num_labels[k]
            else:
                postfix_position_ids += [max(postfix_position_ids) + 1 + i for i in range(len_prompt[k])]
                position_id = max(postfix_position_ids) + 1
                postfix_position_ids += [position_id] * level_num_labels[k]

        # Last one for [SEP] token at the end.
        postfix_position_ids.append(max(postfix_position_ids) + 1)
        padding_position_ids = [0] * len_padding
        position_ids = source_position_ids + padding_position_ids + postfix_position_ids
        data_dict["position_ids"].append(position_ids)

        # Add positions of labels
        positions = []
        for k in range(len(level_num_labels)):
            if k == 0:
                # First level positions
                positions += [len_trunc_text + len_padding + len_prompt[k] + j for j in range(level_num_labels[k])]
            else:
                # Add final positions of previous level (+1) as starting point for next level
                positions += [max(positions) + 1 + len_prompt[k] + j for j in range(level_num_labels[k])]
        data_dict["positions"].append(positions)

        if token_type_setting == 'all_0s':
            token_type_ids = [0] * total_token_len
        elif token_type_setting == 'template_1s':
            token_type_ids = [0] * (len_trunc_text + len_padding) + [1] * len_postfix
        elif token_type_setting == 'classes_1s':
            token_type_ids = [0] * total_token_len
            for i in range(len(positions)):
                token_type_ids[positions[i]] = 1

        data_dict["token_type_ids"].append(token_type_ids)

        # Add one-hot-encoded labels
        one_hot_labels = np.zeros(num_labels)
        one_hot_labels[label] = 1
        data_dict['labels'].append(torch.from_numpy(np.float32(one_hot_labels)))
    return data_dict

# This tokenizes the data by always putting the template in the same position, after the padding. For flat prompts at start of template.
def tokenize_data_template_end_flat(dataset, tokenizer, postfix, level_num_labels, num_labels, depth2label, len_prompt, max_length = 512, text_mask = False):
    data_dict = {"input_ids" : [], "attention_mask" : [], "position_ids": [], "token_type_ids": [], "positions" : [], "labels" : [], "text_length" : []}
    for text, label in zip(dataset['token'], dataset['label']):
        # Add input_ids

        text_tokens = ["[CLS]"] + tokenizer.tokenize(text)

        if add_SEP:
            text_tokens += ["[SEP]"]

        # Check if the text is too long.
        len_text = len(text_tokens)
        len_postfix = len(postfix)
        num_levels = len(level_num_labels)

        # The max length the text tokens can take up, -1 is for the SEP token at end of postfix.
        max_len_text = max_length - num_levels - sum(len_prompt) - 1

        overflow_len = len_text - max_len_text
        if overflow_len > 0:
            text_tokens = text_tokens[:len_text - overflow_len]
        len_trunc_text = len(text_tokens)

        data_dict["text_length"].append(len_trunc_text)

        # Add padding
        len_padding = max_len_text - len_trunc_text
        if len_padding > 0:
            text_tokens += ["[PAD]"] * len_padding

        text_tokens += postfix
        text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        total_token_len = max_len_text + len_postfix
        len_tokens = max_len_text - len_padding
        data_dict["input_ids"].append(text_token_ids)

        # Add attention_mask
        if text_mask:
            # text attention mask
            attention_mask = [[1] * len_trunc_text + [0] * (total_token_len - len_trunc_text)] * len_trunc_text
            # postfix attention mask
            attention_mask += [[1] * len_tokens + [0] * (total_token_len - len_tokens)] * len_postfix
            # padding attention mask
            attention_mask += [[0] * total_token_len] * (total_token_len - len_tokens)
        else:
            attention_mask = [1] * len_trunc_text + [0] * (len_padding) + [1] * len_postfix
        data_dict["attention_mask"].append(attention_mask)

        # Add position_ids
        source_position_ids = [i for i in range(len_trunc_text)]
        
        postfix_position_ids = []
        for k in range(len(level_num_labels)):
            if k == 0:
                postfix_position_ids += [len_trunc_text + len_padding + i for i in range(len_prompt[k])]
            else:
                postfix_position_ids += [max(postfix_position_ids) + 1 + i for i in range(len_prompt[k])]
        position_id = max(postfix_position_ids) + 1
        postfix_position_ids += [position_id] * sum(level_num_labels)

        # Last one for [SEP] token at the end.
        postfix_position_ids.append(max(postfix_position_ids) + 1)
        padding_position_ids = [0] * len_padding
        position_ids = source_position_ids + padding_position_ids + postfix_position_ids
        data_dict["position_ids"].append(position_ids)

        if token_type_setting == 'all_0s':
            token_type_ids = [0] * total_token_len
        elif token_type_setting == 'template_1s':
            token_type_ids = [0] * (len_trunc_text + len_padding) + [1] * len_postfix
        elif token_type_setting == 'classes_1s':
            token_type_ids = [0] * total_token_len
            for i in range(len(positions)):
                token_type_ids[positions[i]] = 1
        data_dict["token_type_ids"].append(token_type_ids)

        # Add positions of labels
        positions = []
        first_position = len_trunc_text + len_padding + sum(len_prompt)
        positions += [first_position + i for i in range(sum(level_num_labels))]
        data_dict["positions"].append(positions)

        # Add one-hot-encoded labels
        one_hot_labels = np.zeros(num_labels)
        one_hot_labels[label] = 1
        data_dict['labels'].append(torch.from_numpy(np.float32(one_hot_labels)))
    return data_dict