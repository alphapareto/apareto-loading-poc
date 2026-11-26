"""Validate selected request controls before native vllm_xargs transport, and attach the server's think-end
ladder to thinking requests."""
import json
import threading

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def request_controls(body, config):
    if not isinstance(body,dict):raise ValueError('Request body must be a JSON object')
    extra=dict(body.get('vllm_xargs') or {})
    for name in ('select','expert_budget'):
        if name in body:
            if name in extra and extra[name]!=body[name]:raise ValueError(f'Conflicting {name} values')
            extra[name]=body.pop(name)
    selected=extra.get('select',True)
    budget=extra.get('expert_budget',config['budget'])
    if selected is not True:raise ValueError('This server selects every request; select=false is not served')
    if budget!=config['budget']:
        raise ValueError(f"This server serves the budget it was started with ({config['budget']}); got {budget!r}")
    if (body.get('chat_template_kwargs') or {}).get('enable_thinking',True) is False:
        raise ValueError('This server serves thinking requests only; enable_thinking=false is not served')
    if isinstance(body.get('prompt'),list):
        prompt=body['prompt']
        if prompt and all(type(token) is int for token in prompt) and len(prompt)<8:
            raise ValueError('Selected prompts require at least eight token IDs')
    body['vllm_xargs']={**extra,'select':selected,'expert_budget':budget}
    return body


_LADDER = {'lock': threading.Lock(), 'path': None, 'spec': None, 'config': None}


def _ladder_config(path, model_config):
    """The engine config built from the ladder spec and the served tokenizer, once per server (the token sets scan
    the whole vocabulary)."""
    with _LADDER['lock']:
        if _LADDER['path'] != path:
            from transformers import AutoTokenizer
            from apareto_loader.ladder_spec import build_config
            spec = json.loads(open(path).read())
            tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer, revision=model_config.tokenizer_revision,
                                                      trust_remote_code=model_config.trust_remote_code)
            # The reserve check needs a generation cap; it is repeated per request against the request's own cap.
            cap = spec['final']['at_tok'] + spec['final_wait'] + spec['min_answer_reserve']
            _LADDER.update(path=path, spec=spec, config=build_config(spec, tokenizer, cap))
        return _LADDER['spec'], _LADDER['config']


def attach_ladder(body, path, vllm_config):
    """A chat request with thinking on (the template's default) gets the ladder in vllm_xargs, unless it already
    carries one or its generation cap leaves less than the spec's answer reserve after the forced think-end."""
    if body.get('messages') is None or 'ladder' in (body.get('vllm_xargs') or {}):
        return body
    if (body.get('chat_template_kwargs') or {}).get('enable_thinking', True) is False:
        return body
    from apareto_loader.ladder_spec import wire
    spec, config = _ladder_config(path, vllm_config.model_config)
    cap = body.get('max_completion_tokens') or body.get('max_tokens') or vllm_config.model_config.max_model_len
    if int(cap) - spec['final']['at_tok'] - spec['final_wait'] < spec['min_answer_reserve']:
        return body
    body['vllm_xargs'] = {**(body.get('vllm_xargs') or {}), 'ladder': wire(config)}
    return body


class BankControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self,request,call_next):
        if request.method=='POST' and request.url.path in ('/v1/completions','/v1/chat/completions'):
            try:
                config=request.app.state.vllm_config.additional_config['apareto']
                body=request_controls(await request.json(),config)
                if config.get('ladder'):
                    body=attach_ladder(body,config['ladder'],request.app.state.vllm_config)
                request._body=json.dumps(body).encode()
            except (ValueError,TypeError) as error:
                return JSONResponse({'error':{'message':str(error),'type':'invalid_request_error'}},status_code=400)
        return await call_next(request)
