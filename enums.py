from enum import Enum


class PromptType(Enum):
    custom = -1
    plain = 0
    instruct = 1
    quality = 2
    human_bot = 3
    dai_faq = 4
    summarize = 5
    simple_instruct = 6
    instruct_vicuna = 7
    instruct_with_end = 8
    human_bot_orig = 9
    prompt_answer = 10
    open_assistant = 11
    wizard_lm = 12
    wizard_mega = 13
    instruct_vicuna2 = 14
    instruct_vicuna3 = 15
    wizard2 = 16
    wizard3 = 17
    instruct_simple = 18
    wizard_vicuna = 19
    openai = 20
    openai_chat = 21
    gptj = 22


class DocumentChoices(Enum):
    All_Relevant = 0
    All_Relevant_Only_Sources = 1
    Only_All_Sources = 2
    Just_LLM = 3


class LangChainMode(Enum):
    """LangChain mode"""

    DISABLED = "Disabled"
    CHAT_LLM = "ChatLLM"
    LLM = "LLM"
    ALL = "All"
    WIKI = "wiki"
    WIKI_FULL = "wiki_full"
    USER_DATA = "UserData"
    MY_DATA = "MyData"
    GITHUB_H2OGPT = "github h2oGPT"
    H2O_DAI_DOCS = "DriverlessAI docs"


no_server_str = no_lora_str = no_model_str = '[None/Remove]'


# from site-packages/langchain/llms/openai.py, but needed since ChatOpenAI doesn't have this information
model_token_mapping = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16*1024,
    "gpt-3.5-turbo-0301": 4096,
    "text-ada-001": 2049,
    "ada": 2049,
    "text-babbage-001": 2040,
    "babbage": 2049,
    "text-curie-001": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}
