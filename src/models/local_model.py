from typing import TypeVar

import torch
import ujson
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base_model import BaseModel

JSON_T = dict | list
JSON_T_VAR = TypeVar("JSON_T_VAR", bound=JSON_T)

class BaichuanOmni1d5Model(BaseModel):
    def __init__(self, model_path: str, temperature: float, do_sample: bool, max_new_tokens: int):
        self.role_prefix = {
            'system': '<B_SYS>',
            'user': '<C_Q>',
            'assistant': '<C_A>',
            'audiogen': '<audiotext_start_baichuan>'
        }
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.training = False
        self.model.bind_processor(self.tokenizer, training=False)
        self.image_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.video_config.image_start_token_id)
        self.image_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.video_config.image_end_token_id)
        print("BaiChuanOmni1d5Model loaded.")

    def generate_from_image_and_text(self, image_path: str, prompt: str, output_type: type[JSON_T_VAR] = dict):
        try:
            Image.open(image_path).close()
        except Exception as e:
            print(f"[ERROR] {e}")
            print(f"[ERROR] {image_path}")
            raise e
        
        image_info = self.image_start_token + ujson.dumps({'local': image_path}, ensure_ascii=False) + self.image_end_token
        content = image_info + prompt
        
        message = self.role_prefix["user"] + content + self.role_prefix["assistant"]
        processed_input = self.model.processor([message])
        plen = processed_input.input_ids.shape[1]
        
        text_output = self.model.generate(
            input_ids=processed_input.input_ids.cuda(),
            attention_mask=processed_input.attention_mask.cuda() if processed_input.attention_mask is not None else None,
            images=[torch.tensor(img, dtype=torch.float32).cuda() for img in processed_input.images] if processed_input.images is not None else None,
            patch_nums=processed_input.patch_nums if processed_input.patch_nums is not None else None,
            images_grid=processed_input.images_grid if processed_input.images_grid is not None else None,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            stop_strings=['<|endoftext|>'],
            do_sample=self.do_sample,
            temperature=self.temperature,
            repetition_penalty=1.1,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id     # Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
        )
        
        new_text = self.tokenizer.decode(text_output.sequences[0, plen:])
        assert (res_text:=new_text.replace('<|endoftext|>', '').strip())
        if output_type is str:
            return res_text
        try:
            res = self.t2j(res_text, output_type)
        except Exception as e:
            raise type(e)(f"[{str(e)}] Original Text: {res_text}")
        return res

    def generate_from_text(self, prompt: str, output_type: type[JSON_T_VAR] = dict):
        """
        LocalModel does not support text-only generation.
        Please use ApiModel for this functionality.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support 'generate_from_text'. "
            "Please use an API-based model instead."
        )
