import json
import os
import random
from collections import defaultdict
import logging
import re
import sys

from guidance import gen, user, assistant

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'household_kb'))

from reibench.planners.alfred_planner import AlfredTaskPlanner
from reibench.envs.alfred.utils import ithor_name_to_natural_word, find_indefinite_article

from household_kb.knowledge_base.knowledge_base import KnowledgeBase
from household_kb.knowledge_base.yaml_loader import load_from_yaml
from household_kb.entities import ObjectCategory
from household_kb.knowledge_base.query_engine import QueryCriteria

log = logging.getLogger(__name__)


class ReactAlfredPlanner(AlfredTaskPlanner):
    REACT_ACTIONS = ['[PLAN]', '[ASK]', '[QUERY]']

    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_react_steps = 6
        self._init_knowledge_base()

    def _init_knowledge_base(self):
        kb_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'household_kb', 'data', 'alfred_objects_new.yaml'
        )
        if os.path.exists(kb_path):
            self.knowledge_base = load_from_yaml(kb_path)
            log.info(f"Knowledge base initialized with {len(self.knowledge_base)} objects")
        else:
            self.knowledge_base = KnowledgeBase()
            log.warning(f"Knowledge base file not found: {kb_path}")

    def init_kb_prompt(self):
        kb_description = """
## Knowledge Base API

You have access to a household knowledge base that stores information about objects in the environment.

### Available Actions:
1. **query_kb** - Query the knowledge base to resolve ambiguities or get object information

### Query Methods:
- `kb.query_subclass_by_category(category)` - Query objects by category (FOOD, UTENSIL, CONTAINER, APPLIANCE, FURNITURE, ELECTRONIC)
- `kb.query_subclass_by_affordance(can_pick=True/False, can_contain=True/False, can_open=True/False, can_toggle=True/False, can_slice=True/False, can_heat=True/False, can_cool=True/False)` - Query objects by affordance properties
- `kb.resolve_vague_reference(reference)` - Resolve vague references like 'fruit', 'container', 'electronic' to concrete objects
- `kb.query_entities_attr(entities, attr_path)` - Query specific attributes (e.g., "physical.weight", "affordance.can_pick")

### Queryable Attributes:
- `name`: Object name (e.g., "Apple", "Microwave")
- `category`: Object category
- `tags`: List of descriptive tags
- `physical.weight`: Object weight in grams
- `physical.material`: Object material
- `affordance.can_pick`: Whether object can be picked up
- `affordance.can_contain`: Whether object can contain things
- `affordance.can_open`: Whether object can be opened
- `affordance.can_toggle`: Whether object can be turned on/off
- `affordance.can_slice`: Whether object can be sliced
- `affordance.can_heat`: Whether object can be heated
- `affordance.can_cool`: Whether object can be cooled

### Usage Example:
- Query all pickable objects: kb.query_subclass_by_affordance(can_pick=True)
- Resolve 'fruit' reference: kb.resolve_vague_reference('fruit')
- Query food category: kb.query_subclass_by_category(ObjectCategory.FOOD)
- Query entities' Attributes: kb.query_entities_attr(['apple', 'pot', 'pen'], physical.weight)

### When to Use:
- when the instruction contains vague references (e.g., 'fruit', 'container', 'electronic device', 'the heavy one')
- when you need to determine specific objects that match certain properties

"""
        return kb_description
    
    def init_react_prompt(self, allow_ask = False):
        react_prompt_noask = """
        You are a robot assistant that uses reasoning and actions to complete household tasks. For each instruction, you should think step by step and select the most appropriate action.
### Available Actions:
1. **[PLAN]** - Generate the next action to execute when you have enough information.
   - Use when: You understand the task and know what action to perform next.
   - Format: [PLAN] <action description>
   - Example: [PLAN] 1. find a ladle, 2. pick up the ladle, 3. find a sink, 4. put down the ladle, 5. done.
2. **[QUERY]** - Query the knowledge base to resolve ambiguities or get object information.
   - Use when: The instruction contains vague references like 'fruit', 'container', 'electronic device', 'the heavier one', or you need to find objects with specific properties.
   - Format: [QUERY] <query api>
   - Example: [QUERY] kb.query_entities_attr(['apple', 'pot'], physical.weight)
"""
        react_prompt_ask = """
You are a robot assistant that uses reasoning and actions to complete household tasks. For each instruction, you should think step by step and select the most appropriate action.
### Available Actions:
1. **[PLAN]** - Generate the next action to execute when you have enough information.
   - Use when: You understand the task and know what action to perform next.
   - Format: [PLAN] <action description>
   - Example: [PLAN] 1. find a ladle, 2. pick up the ladle, 3. find a sink, 4. put down the ladle, 5. done.
2. **[ASK]** - Ask the human for clarification when the instruction is ambiguous or missing required information.
   - Use when: The instruction contains vague references you cannot resolve with available context.
   - Format: [ASK] <question to ask>
   - Example: [ASK] What does it refer to.
3. **[QUERY]** - Query the knowledge base to resolve ambiguities or get object information.
   - Use when: The instruction contains vague references like 'fruit', 'container', 'electronic device', 'the heavier one', or you need to find objects with specific properties.
   - Format: [QUERY] <query api>
   - Example: [QUERY] kb.query_entities_attr(['apple', 'pot'], physical.weight)
### Reasoning Format:
Think step by step. Before taking any action, reason about:
1. What is the user asking me to do?
2. Do I have enough information to proceed?
3. Are there any ambiguous references that need clarification?
"""
        return react_prompt_ask if allow_ask else react_prompt_noask

    def init_prompt(self, cfg):
        kb_prompt = self.init_kb_prompt()
        react_prompt = self.init_react_prompt()
        if cfg.planner.use_predefined_prompt:
            example_prompts = self.load_prompt(cfg)
            return f"{react_prompt}\n{kb_prompt}\n###example###\n{example_prompts}\n"

    def load_prompt(self, cfg):
        _prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        with open(os.path.join(_prompts_dir, "predefined_prompt_react.txt")) as f:
            prompt = f.read()
        return prompt
        
    def parse_react_action(self, action_str):
        action_str = action_str.strip()

        for react_action in self.REACT_ACTIONS:
            if action_str.startswith(react_action):
                return react_action, action_str[len(react_action):].strip()

        # if 'plan' in action_str.lower()[:20]:
        #     return '[PLAN]', action_str[4:].strip() if len(action_str) > 4 else ''
        # elif 'ask' in action_str.lower()[:20]:
        #     return '[ASK]', action_str[3:].strip() if len(action_str) > 3 else ''
        # elif 'query' in action_str.lower()[:20] or 'kb' in action_str.lower()[:20]:
        #     return '[QUERY]', action_str[8:].strip() if len(action_str) > 8 else ''
        # else:
        return '[UNKNOWN ACTION]', action_str
    
    def execute_kb_query(self, query_str):
        query_str = query_str.lower().strip()

        try:
            if 'vague' in query_str or 'reference' in query_str or 'resolve' in query_str:
                ref_match = re.search(r"['\"]([^'\"]+)['\"]", query_str)
                if ref_match:
                    reference = ref_match.group(1)
                    results = self.knowledge_base.resolve_vague_reference(reference)
                    return [obj.name for obj in results]

            if 'query_entities_attr' in query_str or 'entities_attr' in query_str or 'attr' in query_str:
                ids_match = re.search(r"\[\s*([^\]]+)\s*\]", query_str)
                attr_match = re.search(r"(?:physical|state|affordance|name|category|tags)\.[a-z_]+", query_str)

                if ids_match and attr_match:
                    ids_str = ids_match.group(1)
                    entity_ids = [x.strip().strip("'\"") for x in ids_str.split(",")]
                    attr_path = attr_match.group(0)

                    entities = []
                    for eid in entity_ids:
                        entity = self.knowledge_base.get(eid)
                        if entity:
                            entities.append(entity)

                    results = self.knowledge_base.query_entities_attr(entities, attr_path)
                    return [f"{obj.name}: {val}" for obj, val in results]

            if 'category' in query_str:
                for cat in ObjectCategory:
                    if cat.name.lower() in query_str:
                        results = self.knowledge_base.query_subclass_by_category(cat)
                        return [obj.name for obj in results]

            if 'affordance' in query_str or any(kw in query_str for kw in ['pick', 'contain', 'support', 'cut', 'slice', 'heat', 'cool', 'open', 'toggle', 'on', 'off']):
                affordance_params = {}
                param_mapping = {
                    'pick': 'can_pick',
                    'contain': 'can_contain',
                    'support': 'can_support',
                    'cut': 'can_cut',
                    'slice': 'can_slice',
                    'heat': 'can_heat',
                    'cool': 'can_cool',
                    'open': 'can_open',
                    'toggle': 'can_toggle',
                }

                for keyword, param in param_mapping.items():
                    match = re.search(rf"{param}\s*=\s*(true|false)", query_str)
                    if match:
                        affordance_params[param] = match.group(1) == 'true'

                if affordance_params:
                    results = self.knowledge_base.query_subclass_by_affordance(**affordance_params)
                    return [obj.name for obj in results]

                if 'pick' in query_str:
                    results = self.knowledge_base.query_subclass_by_affordance(can_pick=True)
                    return [obj.name for obj in results]
                if 'contain' in query_str:
                    results = self.knowledge_base.query_subclass_by_affordance(can_contain=True)
                    return [obj.name for obj in results]
                if 'open' in query_str:
                    results = self.knowledge_base.query_subclass_by_affordance(can_open=True)
                    return [obj.name for obj in results]
                if 'toggle' in query_str or 'on' in query_str or 'off' in query_str:
                    results = self.knowledge_base.query_subclass_by_affordance(can_toggle=True)
                    return [obj.name for obj in results]
                if 'slice' in query_str or 'cut' in query_str:
                    results = self.knowledge_base.query_subclass_by_affordance(can_cut=True)
                    return [obj.name for obj in results]
                if 'heat' in query_str:
                    results = self.knowledge_base.query_subclass_by_affordance(can_heat=True)
                    return [obj.name for obj in results]
                if 'cool' in query_str or 'cold' in query_str or 'fridge' in query_str:
                    results = self.knowledge_base.query_subclass_by_affordance(can_cool=True)
                    return [obj.name for obj in results]

            if 'food' in query_str or 'fruit' in query_str or 'vegetable' in query_str:
                results = self.knowledge_base.query_subclass_by_category(ObjectCategory.FOOD)
                return [obj.name for obj in results]
            if 'utensil' in query_str or 'tool' in query_str:
                results = self.knowledge_base.query_subclass_by_category(ObjectCategory.UTENSIL)
                return [obj.name for obj in results]

            return ["Query not understood, please rephrase"]

        except Exception as e:
            log.warning(f"KB query error: {e}")
            return ["Error in query execution"]

    def react_step(self, query, prompt, reasoning_history=None):
        if reasoning_history is None:
            reasoning_history = []
        
        if "MiniMax" in self.model_name:
            with user():
                lm = self.planner_model + prompt
            with assistant():
                lm += gen("response", temperature=0.3, max_tokens=1024)
        else:
            lm = self.planner_model + prompt + gen("response", temperature=0, max_tokens=1024)
        reasoning = lm['response']
        if "MiniMax" in self.model_name:
            reasoning = re.sub(r'<think>.*?</think>', '', reasoning, flags=re.DOTALL)
        reasoning = reasoning.replace('Robot: ', '').strip()
        log.info(f"react step:{reasoning}")
        reasoning_history.append(reasoning)

        action_type, action_detail = self.parse_react_action(reasoning)

        if action_type == '[QUERY]':
            kb_results = self.execute_kb_query(action_detail)
            observation = f"Observation: KB returned {len(kb_results)} objects: {', '.join(kb_results)}"
            reasoning_history.append(observation)
            return 'continue', prompt, reasoning_history

        if action_type == '[ASK]':
            return f"ask: {action_detail}", prompt, reasoning_history
        
        if action_type == '[PLAN]':
            print("[plan mode]")
            return f'plan: {action_detail}', prompt, reasoning_history
        
        error_action_prompt = "The response does not start with [PLAN] or [QUERY]"
        reasoning_history.append(error_action_prompt)
        return 'continue', prompt, reasoning_history
        

    def plan_whole(self, query, prev_steps=..., prev_msgs=...):
        step_seq = []
        skill_set_size_seq = []

        skills_text = ', '.join([x.strip() for x in self.skill_set])
        prompt = self.prompt
        prompt += f"""
        ### Restriction
        if you choose [PLAN] action
        You should **only** use actions of this list: {skills_text}.
        You should **only** use actions of the upper list.
        You should **only** use actions of the upper list.
        """ 
        prompt += f"\nHuman instruction: {query.strip()}\n"


        prompt += "\nYour response ([PLAN]: ... / [QUERY]: ...):\n"
        history = None
        for i in range(self.max_react_steps):
            if history:
                prompt += "## Reasoning History:\n"
                for reason in history[-5:]:
                    prompt += reason + "\n"
            answer, prompt, history = self.react_step(query, prompt, history)
            if answer == 'continue':
                continue
            elif answer.startswith('ask'):
                assert 0, "not support ask"
            elif answer.startswith('plan'):
                break
        
        answer = answer.replace('plan: ', '')
        actions = [action.strip(' 1234567890.') for action in answer.split(',')]
        step_seq = actions
        return step_seq, skill_set_size_seq

