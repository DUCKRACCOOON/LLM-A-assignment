"""Tiny random-model infrastructure check, never a paper experiment."""
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, pipeline


def main():
    vocab = {'<unk>': 0, '<bos>': 1, '<eos>': 2, 'hello': 3, 'world': 4, 'path': 5, 'start': 6, 'goal': 7}
    backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                       bos_token='<bos>', eos_token='<eos>', pad_token='<eos>')
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=8, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=2)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).to(device='cuda:0', dtype=torch.bfloat16).eval()
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
    prompt = 'hello world'
    ids = pipe.preprocess(prompt)['input_ids'][0].tolist()
    result = pipe(prompt, max_new_tokens=4, do_sample=False, temperature=None, top_p=None,
                  eos_token_id=[2], pad_token_id=2, return_tensors=True)
    tokens = result[0]['generated_token_ids']
    assert tokens[:len(ids)] == ids
    assert 1 <= len(tokens) - len(ids) <= 4
    # Check our response decoding against the original author's pipeline call.
    reference = pipe(prompt, max_new_tokens=4, do_sample=False, temperature=None, top_p=None,
                     eos_token_id=[2], pad_token_id=2)[0]['generated_text'][len(prompt):]
    full = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    prefix = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assert full[len(prefix):] == reference
    record = {'purpose': 'infrastructure only; random tiny weights; NOT paper results',
              'passed': True, 'device': torch.cuda.get_device_name(0), 'dtype': str(model.dtype),
              'generated_tokens': len(tokens) - len(ids), 'pipeline_decode_matches_author': True}
    out = Path(__file__).resolve().parents[1] / 'results/llama/infrastructure_smoke.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
