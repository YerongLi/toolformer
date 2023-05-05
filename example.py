import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from toolformer.data_generator import DataGenerator
from toolformer.api import CalculatorAPI
from toolformer.prompt import calculator_prompt
from toolformer.utils import yaml2dict


# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# # create formatter
# formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")

# # add formatter to ch
# ch.setFormatter(formatter)

# logging.info(f'Logger start: {os.uname()[1]}')
# print('Start logging')

import logging

# create logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    filename='./output.log',
    datefmt='%Y-%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

# quit()
config = yaml2dict('./configs/default.yaml')
calculator_api = CalculatorAPI(
    "Calculator", calculator_prompt,
    sampling_threshold=0.2, filtering_threshold=0.2
)

# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

logging.info('Finishing loading from AutoModelForCausalLM')
text = "From this, we have 10 - 5 minutes = 5 minutes."
apis = [calculator_api]
generator = DataGenerator(config, model, tokenizer, apis=apis)

augumented_text_ids = generator.generate(text)

print(tokenizer.decode(augumented_text_ids[0][0], skip_special_tokens=True))