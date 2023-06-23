import os
import subprocess
import time
from datetime import datetime

import pytest

from client_test import run_client_many
from enums import PromptType
from tests.test_langchain_units import have_openai_key
from tests.utils import wrap_test_forked


@wrap_test_forked
@pytest.mark.parametrize("base_model",
                         ['h2oai/h2ogpt-oig-oasst1-512-6_9b',
                          'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',
                          'llama', 'gptj']
                         )
def test_gradio_inference_server(base_model,
                                 prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                 langchain_mode='Disabled', user_path=None,
                                 visible_langchain_modes=['UserData', 'MyData'],
                                 reverse_docs=True):
    if base_model in ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-oasst1-512-12b']:
        prompt_type = PromptType.human_bot.name
    elif base_model in ['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2']:
        prompt_type = PromptType.prompt_answer.name
    elif base_model in ['llama']:
        prompt_type = PromptType.wizard2.name
    elif base_model in ['gptj']:
        prompt_type = PromptType.gptj.name
    else:
        raise NotImplementedError(base_model)

    main_kwargs = dict(base_model=base_model, prompt_type=prompt_type, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode, user_path=user_path,
                       visible_langchain_modes=visible_langchain_modes,
                       reverse_docs=reverse_docs)

    # inference server
    inf_port = os.environ['GRADIO_SERVER_PORT'] = "7860"
    from generate import main
    main(**main_kwargs)

    # server that consumes inference server
    client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
    from generate import main
    main(**main_kwargs, inference_server=f'http://127.0.0.1:{inf_port}')

    # client test to server that only consumes inference server
    from client_test import run_client_chat
    os.environ['HOST'] = f"http://127.0.0.1:{client_port}"
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''

    # will use HOST from above
    ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)  # client shouldn't have to specify
    if base_model == 'gptj':
        assert 'I am a bot.' in ret1['response'] or 'can I assist you today?' in ret1[
            'response'] or 'I am a student at' in ret1['response'] or 'am a person who' in ret1['response'] or 'I am' in \
               ret1['response'] or "I'm a student at" in ret1['response']
        assert 'Birds' in ret2['response'] or 'Once upon a time' in ret2['response']
        assert 'Birds' in ret3['response'] or 'Once upon a time' in ret3['response']
        assert 'I am a bot.' in ret4['response'] or 'can I assist you today?' in ret4[
            'response'] or 'I am a student at' in ret4['response'] or 'am a person who' in ret4['response'] or 'I am' in \
               ret4['response'] or "I'm a student at" in ret4['response']
        assert 'I am a bot.' in ret5['response'] or 'can I assist you today?' in ret5[
            'response'] or 'I am a student at' in ret5['response'] or 'am a person who' in ret5['response'] or 'I am' in \
               ret5['response'] or "I'm a student at" in ret5['response']
        assert 'I am a bot.' in ret6['response'] or 'can I assist you today?' in ret6[
            'response'] or 'I am a student at' in ret6['response'] or 'am a person who' in ret6['response'] or 'I am' in \
               ret6['response'] or "I'm a student at" in ret6['response']
        assert 'I am a bot.' in ret7['response'] or 'can I assist you today?' in ret7[
            'response'] or 'I am a student at' in ret7['response'] or 'am a person who' in ret7['response'] or 'I am' in \
               ret7['response'] or "I'm a student at" in ret7['response']
    elif base_model == 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2':
        assert 'I am a language model trained' in ret1['response'] or 'I am an AI language model developed by' in \
               ret1['response'] or 'I am a chatbot.' in ret1['response']
        assert 'Once upon a time' in ret2['response']
        assert 'Once upon a time' in ret3['response']
        assert 'I am a language model trained' in ret4['response'] or 'I am an AI language model developed by' in \
               ret4['response'] or 'I am a chatbot.' in ret4['response']
        assert 'I am a language model trained' in ret5['response'] or 'I am an AI language model developed by' in \
               ret5['response'] or 'I am a chatbot.' in ret5['response']
        assert 'I am a language model trained' in ret6['response'] or 'I am an AI language model developed by' in \
               ret6['response'] or 'I am a chatbot.' in ret6['response']
        assert 'I am a language model trained' in ret7['response'] or 'I am an AI language model developed by' in \
               ret7['response'] or 'I am a chatbot.' in ret7['response']
    elif base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        assert 'h2oGPT' in ret1['response']
        assert 'Birds' in ret2['response']
        assert 'Birds' in ret3['response']
        assert 'h2oGPT' in ret4['response']
        assert 'h2oGPT' in ret5['response']
        assert 'h2oGPT' in ret6['response']
        assert 'h2oGPT' in ret7['response']
    elif base_model == 'llama':
        assert 'I am a bot.' in ret1['response'] or 'can I assist you today?' in ret1['response']
        assert 'Birds' in ret2['response'] or 'Once upon a time' in ret2['response']
        assert 'Birds' in ret3['response'] or 'Once upon a time' in ret3['response']
        assert 'I am a bot.' in ret4['response'] or 'can I assist you today?' in ret4['response']
        assert 'I am a bot.' in ret5['response'] or 'can I assist you today?' in ret5['response']
        assert 'I am a bot.' in ret6['response'] or 'can I assist you today?' in ret6['response']
        assert 'I am a bot.' in ret7['response'] or 'can I assist you today?' in ret7['response']
    print("DONE", flush=True)


def run_docker(inf_port, base_model):
    datetime_str = str(datetime.now()).replace(" ", "_").replace(":", "_")
    msg = f"Starting HF inference {datetime_str}..."
    print(msg, flush=True)
    home_dir = os.path.expanduser('~')
    data_dir = f'{home_dir}/.cache/huggingface/hub/'
    cmd = ["docker"] + [
        'run',
        '--gpus',
        'device=0',
        '--shm-size',
        '1g',
        '-e',
        'TRANSFORMERS_CACHE="/.cache/"',
        '-p',
        f'{inf_port}:80',
        '-v',
        f'{home_dir}/.cache:/.cache/',
        '-v',
        f'{data_dir}:/data',
        'ghcr.io/huggingface/text-generation-inference:0.8.2',
        '--model-id',
        base_model,
        '--max-input-length',
        '2048',
        '--max-total-tokens',
        '3072',
    ]
    print(cmd, flush=True)
    p = subprocess.Popen(cmd,
                         stdout=None, stderr=subprocess.STDOUT,
                         )
    print("Done starting autoviz server", flush=True)
    return p.pid


@wrap_test_forked
@pytest.mark.parametrize("base_model",
                         # FIXME: Can't get 6.9 or 12b (quantized or not) to work on home system, so do falcon only for now
                         # ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2']
                         ['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2']
                         )
def test_hf_inference_server(base_model,
                             prompt='Who are you?', stream_output=False, max_new_tokens=256,
                             langchain_mode='Disabled', user_path=None,
                             visible_langchain_modes=['UserData', 'MyData'],
                             reverse_docs=True):
    if base_model in ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-oasst1-512-12b']:
        prompt_type = PromptType.human_bot.name
    else:
        prompt_type = PromptType.prompt_answer.name
    main_kwargs = dict(base_model=base_model, prompt_type=prompt_type, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode, user_path=user_path,
                       visible_langchain_modes=visible_langchain_modes,
                       reverse_docs=reverse_docs)

    # HF inference server
    inf_port = "6112"
    inf_pid = run_docker(inf_port, base_model)
    time.sleep(60)

    try:
        # server that consumes inference server
        client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
        from generate import main
        main(**main_kwargs, inference_server=f'http://127.0.0.1:{inf_port}')

        # client test to server that only consumes inference server
        from client_test import run_client_chat
        os.environ['HOST'] = f"http://127.0.0.1:{client_port}"
        res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                           max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)
        assert res_dict['prompt'] == prompt
        assert res_dict['iinput'] == ''

        # will use HOST from above
        ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)  # client shouldn't have to specify
        # here docker started with falcon before personalization
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            assert 'h2oGPT' in ret1['response']
            assert 'Birds' in ret2['response']
            assert 'Birds' in ret3['response']
            assert 'h2oGPT' in ret4['response']
            assert 'h2oGPT' in ret5['response']
            assert 'h2oGPT' in ret6['response']
            assert 'h2oGPT' in ret7['response']
        else:
            assert 'I am a language model trained' in ret1['response'] or 'I am an AI language model developed by' in \
                   ret1['response']
            assert 'Once upon a time' in ret2['response']
            assert 'Once upon a time' in ret3['response']
            assert 'I am a language model trained' in ret4['response'] or 'I am an AI language model developed by' in \
                   ret4['response']
            assert 'I am a language model trained' in ret5['response'] or 'I am an AI language model developed by' in \
                   ret5['response']
            assert 'I am a language model trained' in ret6['response'] or 'I am an AI language model developed by' in \
                   ret6['response']
            assert 'I am a language model trained' in ret7['response'] or 'I am an AI language model developed by' in \
                   ret7['response']
        print("DONE", flush=True)
    finally:
        # take down docker server
        import signal
        os.kill(inf_pid, signal.SIGTERM)
        os.kill(inf_pid, signal.SIGKILL)

        os.system("docker ps | grep text-generation-inference | awk '{print $1}' | xargs docker stop ")


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@wrap_test_forked
def test_openai_inference_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                 base_model='gpt-3.5-turbo',
                                 langchain_mode='Disabled', user_path=None,
                                 visible_langchain_modes=['UserData', 'MyData'],
                                 reverse_docs=True):
    main_kwargs = dict(base_model=base_model, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode, user_path=user_path,
                       visible_langchain_modes=visible_langchain_modes,
                       reverse_docs=reverse_docs)

    # server that consumes inference server
    client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
    from generate import main
    main(**main_kwargs, inference_server='openai_chat')

    # client test to server that only consumes inference server
    from client_test import run_client_chat
    os.environ['HOST'] = f"http://127.0.0.1:{client_port}"
    res_dict, client = run_client_chat(prompt=prompt, prompt_type='openai_chat', stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''

    # will use HOST from above
    ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)  # client shouldn't have to specify
    assert 'I am an AI language model' in ret1['response']
    assert 'Once upon a time, in a far-off land,' in ret2['response'] or 'Once upon a time' in ret2['response']
    assert 'Once upon a time, in a far-off land,' in ret3['response'] or 'Once upon a time' in ret3['response']
    assert 'I am an AI language model' in ret4['response']
    assert 'I am an AI language model' in ret5['response']
    assert 'I am an AI language model' in ret6['response']
    assert 'I am an AI language model' in ret7['response']
    print("DONE", flush=True)
