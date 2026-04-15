import openai
import torch
import torch._dynamo
torch._dynamo.config.verbose = True
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModel
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.nn import CrossEntropyLoss
import guidance
from guidance import models, select, gen, user, assistant
import logging
import re
import os
import time
from reibench.utils.config_mapper import (
    get_planner_framework, get_model_name, get_prompting_method, get_model_config
)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

log = logging.getLogger(__name__)

def measure_tokens_and_latency(self, prompt, output_text):
    if self.tokenizer is not None:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        input_tokens = input_ids.shape[-1]
    else:
        input_tokens = -1
    if self.tokenizer is not None:
        output_ids = self.tokenizer(output_text, return_tensors="pt")["input_ids"].to(self.device)
        output_tokens = output_ids.shape[-1]
    else:
        output_tokens = -1

    return input_tokens, output_tokens, input_tokens + output_tokens

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_token_id in input_ids[0].tolist()

class TaskPlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = cfg.planner.max_steps
        self.planner_framework = get_planner_framework(cfg)
        model_config = get_model_config(cfg)
        self.model_name = model_config.get('model_name', get_model_name(cfg))
        method_config = get_prompting_method(cfg)
        
        self.scoring_batch_size = cfg.planner.scoring_batch_size
        self.score_function = cfg.planner.score_function
        self.scoring_mode = cfg.planner.scoring_mode
        self.use_predefined_prompt = cfg.planner.use_predefined_prompt
        self.AP = method_config.get('aware_hint', False)
        self.COT = method_config.get('COT', False)
        self.TOCC = method_config.get('TOCC', False)
        self.TOCC_referring_hint = ""
        self.tokenizer = None

        if self.planner_framework == "saycan" or self.planner_framework == "react":
            model_args = {'pretrained_model_name_or_path': self.model_name, 'trust_remote_code': True,
                        'torch_dtype': torch.float16}
            use_accelerate = model_config.get('use_accelerate_device_map', 
                                             getattr(cfg.planner, 'use_accelerate_device_map', True))
            if use_accelerate:
                model_args['device_map'] = "auto"
                load_in_8bit = model_config.get('load_in_8bit', 
                                               getattr(cfg.planner, 'load_in_8bit', False))
                if load_in_8bit:
                    model_args['load_in_8bit'] = True
            hf_auth_token = model_config.get('hf_auth_token', 
                                            getattr(cfg.planner, 'hf_auth_token', ''))
            model_args['use_auth_token'] = hf_auth_token
            
            if cfg.planner.scoring_mode == 'guidance':
                model_args.pop('pretrained_model_name_or_path')
                if "gpt" in self.model_name:
                    openai_model_name = self.model_name
                    openai_api_key = model_config.get('openai_api_key',
                                                     getattr(cfg.planner, 'openai_api_key', ''))
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    self.planner_model = models.OpenAIChat(openai_model_name)
                elif "MiniMax" in self.model_name:
                    import tiktoken
                    openai_model_name = self.model_name
                    os.environ["OPENAI_API_KEY"] = os.environ["MINIMAX_API_KEY"]
                    tokenizer = tiktoken.get_encoding('cl100k_base')
                    self.planner_model = models.OpenAIChat(openai_model_name, base_url="https://api.minimaxi.com/v1", tokenizer=tokenizer)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, force_download=True)
                    if "meta-llama" in self.model_name or 'mistralai' in self.model_name or 'deepseek' in self.model_name or "Qwen2.5" in self.model_name or "qwen2.5" in self.model_name:
                        self.planner_model = models.Transformers(
                            self.model_name,
                            self.tokenizer,
                            device=self.device,
                            torch_dtype=torch.float16,
                        )
                    elif 'gemma' in self.model_name:
                        self.planner_model = models.Transformers(self.model_name, self.tokenizer, device=self.device, torch_dtype=torch.bfloat16)
                logging.getLogger("guidance").setLevel(logging.WARNING)
            elif cfg.planner.scoring_mode == 'api':
                if "MiniMax" in self.model_name:
                    self.base_url = "https://api.minimaxi.com/v1/"
                    self.api_key = os.environ.get("MINIMAX_API_KEY", "")
                openai.base_url = self.base_url
                openai.api_key = self.api_key
            else:
                if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name: 
                    self.model = LlamaForCausalLM.from_pretrained(**model_args)
                    self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(**model_args)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                if not use_accelerate:
                    self.model = self.model.to(self.device)
                self.model.eval()
                self.tokenizer.pad_token_id = 0

            self.prompt = self.init_prompt(cfg)
        else:
            raise ValueError(f"Unsupported planner_framework: {self.planner_framework}. Only 'saycan' and 'react' are supported.")

    def reset(self, nl_act_list, nl_obj_list):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)

    def reset(self):
        self.skill_set = self.init_skill_set()

    def init_prompt(self, cfg):
        raise NotImplementedError()

    def init_skill_set(self, nl_act_list, nl_obj_list):
        raise NotImplementedError()

    def update_skill_set(self, previous_step, nl_obj_list):
        raise NotImplementedError()
    
    
    def count_tokens(self, prompt: str, output_text: str):
        if self.tokenizer is None:
            return -1, -1, -1
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            input_tokens = input_ids.shape[-1]
        except:
            input_tokens = -1
        try:
            output_ids = self.tokenizer(output_text, return_tensors="pt")["input_ids"]
            output_tokens = output_ids.shape[-1]
        except:
            output_tokens = -1

        return input_tokens, output_tokens, input_tokens + output_tokens


    def score(self, prompt, skill_set):
        scores = {}
        batch_skill_set_list = [skill_set[chunk:chunk + self.scoring_batch_size] for chunk in
                                range(0, len(skill_set), self.scoring_batch_size)]
        torch.cuda.empty_cache()
        if self.scoring_mode == 'guidance':
            if "meta-llama" in self.model_name or 'Qwen2.5' in self.model_name or 'mistralai' in self.model_name or 'gemma' in self.model_name or 'deepseek' in self.model_name or 'qwen2.5' in self.model_name:
                out = self.planner_model + prompt + select(skill_set, name='best')
                beststep = out['best']
            elif "gpt" in self.model_name:
                out = self.planner_model + prompt + select(skill_set, name='best')
                beststep = out['best']
            else:
                out = self.guidance_program(prompt=prompt, candidates=skill_set)
            scores = out['score']
        elif self.scoring_mode == 'api':
            assert 0, "step by step not support api"
        else:
            assert False, 'unknown scoring mode'
        return beststep

    def plan_whole(self, query, prev_steps=(), prev_msgs=()):
        step_seq = []
        skill_set_size_seq = []

        prompt_lines = self.prompt.split('\n')
        prompt_examples = prompt_lines[2:]
        example_text = '\n'.join(prompt_examples)
        skills_text = ', '.join([x.strip() for x in self.skill_set])

        previous_plan = ""
        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                previous_plan += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                previous_plan += step + f', {i + 2}. '
        prompt_text = f"""You are a robot operating in a home. A human user can ask you to do various tasks and you are supposed to tell the sequence of actions you would do to accomplish your task.
Examples of human instructions and possible your (robot) answers:{example_text}

Now please answer the sequence of actions for the input instruction.
You should **only** use one of actions of this list: {skills_text}.
You should **only** use one of actions of the upper list.
You should **only** use one of actions of the upper list.
The content in 'Human Previous Inquiry' pertains to previous tasks, and I am not required to complete them.
The 'Human Pending Instruction' section contains the instructions I need to follow and complete.
Input user instruction:
{query}"""
        if len(prev_steps) == 0:
            if self.TOCC:
                self.tocc_answer = self.tocc_step(query)
                prompt_text += f"\nHuman:{self.tocc_answer}"
            
            prompt_text += "\nRobot:"
        else:
            fail_prompt = f"""
Your previous plan was unsuccessful. Here are your plan and the reasons for the failure {previous_plan}.
List the actions with comma seperator again.
Robot:"""
            if self.TOCC:
                prompt_text += f"\nHuman:{self.tocc_answer}"
            prompt_text += f"\n{fail_prompt}"
        if self.scoring_mode == 'api':
            # Direct API call mode - use chat completions API
            answer = self.llm_api_call(prompt_text)
        else:
            with user():
                lm = self.planner_model + prompt_text
            with assistant():
                lm += gen("answer", temperature=0)
            answer = lm['answer']
        if "MiniMax" in self.model_name:
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        answer = answer.strip()
        answer = answer.replace('Robot: ', '')
        actions = [action.strip(' 1234567890.') for action in answer.split(',')]
        step_seq = actions
        return step_seq, skill_set_size_seq
    
    def tocc_step(self, query):
        TOCC_hint = f"""
                    Human pending instruction may contain vague referring expressions, such as ``electronic devices'', ``beverages'', ``fruits'', and ``containers'', which are not specific items. \n
                    You are a robot, your task is to make the `Human Pending Instruction" clear. \n
                    Do not add extra commentary or conversation or the hole plan, only output the clear instruction. \n
                    Use the previous context below to resolve the referring expressions:\n
                    Previous context:\n
                    {query.strip()}\n
                    Please make the `Human Pending Instruction" clear:"""
        answer = self.llm_api_call(TOCC_hint, max_token=1024)
        if "MiniMax" in self.model_name:
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        return answer

    def llm_api_call(self, prompt, max_token=4096):
        messages = [{"role": "user", "content": prompt}]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.,
                    max_tokens=max_token
                )
                reasoning = response.choices[0].message.content.strip()
                break
            except openai.InternalServerError as e:
                if attempt < max_retries - 1:
                    log.warning(f"OpenAI API overloaded, retrying (attempt {attempt + 1}/{max_retries})")
                    continue
                raise 
        answer = response.choices[0].message.content.strip()
        return answer

    def plan_step_by_step(self, query, prev_steps=(), prev_msgs=(), wrong_steps=(), wrong_action_msg=(), reference=None):
        if len(prev_steps) >= self.max_steps:
            return None, None

        prompt = self.prompt + f'{query.strip()}\nRobot: 1. '

        if self.AP == True:
            prompt = self.prompt + f'Robot: {self.AP}\n' + f'{query.strip()}\nRobot: 1. '
        elif self.TOCC == True:
            if len(prev_steps) == 0: 
                TOCC_hint = f"""
                    Human pending instruction may contain vague referring expressions, such as ``electronic devices'', ``beverages'', ``fruits'', and ``containers'', which are not specific items. \n
                    You are a robot, your task is to make the `Human Pending Instruction" clear. \n
                    Do not add extra commentary or conversation or the hole plan, only output the clear instruction. \n
                    Use the previous context below to resolve the referring expressions:\n
                    Previous context:\n
                    {query.strip()}\n
                    Please make the `Human Pending Instruction" clear:"""
                try:
                    prompt_reference = self.planner_model + f"{TOCC_hint}\n" + gen(stop='.')
                    prompt_references = str(prompt_reference).split('\n')
                except AssertionError:
                    raw = self._generate_fallback(f"{TOCC_hint}\n", max_new_tokens=150)
                    if raw is None:
                        raise
                    if '.' in raw:
                        raw = raw.split('.', 1)[0] + '.'
                    prompt_references = raw.split('\n')
                self.prompt_reference = prompt_references[-1].strip()
                with open("output.txt", "a") as f:
                    f.write(f"{str(self.prompt_reference)}\n")
                prompt = self.prompt + f"Human:{self.prompt_reference}\nRobot: 1. "
            else: 
                prompt = self.prompt + f"Human:{self.prompt_reference}\nRobot: 1. "
        elif self.COT == True:
            if len(prev_steps) == 0: 
                cot_hint = self.prompt + f'{query.strip()}\n' \
                + """Human pending instruction may contain some descriptive referring expressions, such as ``electronic devices",  ``beverages", ``fruits", and ``containers", which cannot be identified as specific items. After identifying these referring expressions, I can determine what the referring expressions of these models are from the previous context.\
                The previous context:
                {query.strip()}\n\
                Please answer:\
                What is the object referred to by the referring expression in the ``Human Pending Instruction"?"""
                try:
                    prompt_reference = self.planner_model + f"{cot_hint}\n" + gen(stop=".")
                    prompt_reference = str(prompt_reference).split('\n')
                except AssertionError:
                    raw = self._generate_fallback(f"{cot_hint}\n", max_new_tokens=150)
                    if raw is None:
                        raise
                    if '.' in raw:
                        raw = raw.split('.', 1)[0] + '.'
                    prompt_reference = raw.split('\n')
                self.prompt_reference = prompt_reference[-1] 
                prompt = self.prompt + f'{query.strip()}\n' + f"Hint: {self.prompt_reference}\nRobot: 1. "
            else: 
                prompt = self.prompt + f'{query.strip()}\n' + f"{self.prompt_reference}\nRobot: 1. "

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                prompt += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                prompt += step + f', {i + 2}. '

        
        best_step = self.score(prompt, self.skill_set)
        best_step = best_step.strip()
        return best_step, prompt

    def _generate_fallback(self, prompt_text, max_new_tokens=200):
        if not (hasattr(self, 'planner_model') and hasattr(self.planner_model, 'model_obj') and self.tokenizer is not None):
            return None
        model = self.planner_model.model_obj
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        if getattr(self.tokenizer, 'pad_token_id', None) is not None:
            gen_kw['pad_token_id'] = self.tokenizer.pad_token_id
        elif getattr(self.tokenizer, 'eos_token_id', None) is not None:
            gen_kw['pad_token_id'] = self.tokenizer.eos_token_id
        out = model.generate(**inputs, **gen_kw)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = out[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def duplicate_past_key_values(self, past_key_values, batch_size):
        batch_past_key_values = []
        for layer in range(len(past_key_values)):
            batch_past_key_values_layer = []
            for kv in range(len(past_key_values[layer])):
                batch_past_key_values_layer.append(past_key_values[layer][kv].repeat(batch_size, 1, 1, 1))
            batch_past_key_values_layer = tuple(batch_past_key_values_layer)
            batch_past_key_values.append(batch_past_key_values_layer)
        batch_past_key_values = tuple(batch_past_key_values)
        return batch_past_key_values
