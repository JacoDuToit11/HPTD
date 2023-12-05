# Some constants used in the model

num_soft_prompts = 4

random_prompt = False
# random_prompt = True

# add_SEP = False
add_SEP = True

# token_type_setting = 'all_0s'
token_type_setting = 'template_1s'
# token_type_setting = 'classes_1s'

data_path = '../../Data'

model_type = 'electra'
# model_type = 'deberta'

# frozen = True
frozen = False

# size = 'xsmall'
# size = 'small'
size = 'base'

if model_type == 'electra':
    from HDPTModel import PromptModel
    model_name = 'google/electra-' + size + '-discriminator'
elif model_type == 'deberta':
    from HDPTModelDeBERTa import PromptModel
    model_name = 'microsoft/deberta-v3-' + size

generator_name = 'google/electra-' + size + '-generator'