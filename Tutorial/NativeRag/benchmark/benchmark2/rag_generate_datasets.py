from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import csv
import os
import datasets
from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import pandas as pd
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument



# ====================== config ================================
# export OLLAMA_MODELS="/newdata/HTY/BIO_LLM/DS_70b"
# export HF_ENDPOINT=https://hf-mirror.com
# 设置OLLAMA_MODELS环境变量
pd.set_option("display.max_colwidth", None)
os.environ['OLLAMA_MODELS'] = "/newdata/HTY/BIO_LLM/DS_70b/models"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
pd.set_option("display.max_colwidth", None)
custom_path = "/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/eval_dataset"  # 自定义路径
client = OpenAI(api_key="sk-adefa1e8b6ed4075add4c8c24d570c3b", base_url="https://api.deepseek.com")
llm = Ollama(model="deepseek-r1:1.5b")
# ====================== config ================================



# 选择生成RAG测试集模板
def template_en_or_zh(language):
    QA_generation_prompt = None

    if language == 'zh':
        QA_generation_prompt = """
            您的任务是根据上下文撰写一个事实型问题及其答案。
            事实型问题应能通过上下文中具体、简洁的事实信息来回答。
            问题需模仿用户在搜索引擎中提问的风格，请勿出现“根据上下文”或“文章中提到”等表述。

            请按以下格式输出：

            输出示例:::
            事实型问题：（您的事实型问题）
            答案：（针对该问题的答案）
            以下是提供的上下文：

            上下文：{context}\n
            输出示例:::
        """
    elif language == 'en':
        QA_generation_prompt = """
            Your task is to write a factoid question and an answer given a context.
            Your factoid question should be answerable with a specific, concise piece of factual information from the context.
            Your factoid question should be formulated in the same style as questions users could ask in a search engine.
            This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

            Provide your answer as follows:

            Output:::
            Factoid question: (your factoid question)
            Answer: (your answer to the factoid question)

            Now here is the context.

            Context: {context}\n
            Output:::
        """
    else:
        raise ValueError("QA_generation_prompt failed")

    assert QA_generation_prompt is not None
    return QA_generation_prompt

# 选择测试的模型-可自行扩展 [Deepseek]
def call_llm_API(llm_model, sampled_context, QA_generation_prompt=None):
    if QA_generation_prompt is None:
        QA_generation_prompt = '{context}'
    prompt = QA_generation_prompt.format(context=sampled_context)
    messages=[{"role": "system", "content": ""},
            {"role": "user", "content": f"{prompt}"},]

    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

def call_llm_ollama(llm_model="deepseek-r1:1.5b", sampled_context="", template=None):
    if template is None:
        template = '{context}'
    context = sampled_context
    # 创建RAG Prompt模板
    QA_PROMPT = PromptTemplate(
        input_variables=["context"],
        template=template
    )
    analysis_chain = LLMChain(
        llm=llm,
        prompt=QA_PROMPT,
        verbose=True
    )
    response = analysis_chain.invoke({"context": context})
    return response["text"]

# 准备源数据-可自行扩展
def dataset_prepare():
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train",cache_dir=custom_path)
    langchain_docs = [
        # 可自行扩展
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(ds)
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs_processed = []
    for doc in langchain_docs:
        docs_processed += text_splitter.split_documents([doc])
    
    """
        return list[doc]
        doc -> class 'langchain_core.documents.base.Document'
        doc key-value -> 
            page_content: '';
            metadata : {
                'source': xxx,
            }
    """
    return docs_processed

# 生成对话
def Generate_QA_couple(llm_model, call_model, docs_processed, language, template=None):
    import random
    N_GENERATIONS = 1
    outputs = []
    assert llm_model is not None
    for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS)):
        # Generate QA couple
        output_QA_couple = call_model(llm_model, sampled_context.page_content,template)
        try:
            if language == 'en':
                question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
                answer = output_QA_couple.split("Answer: ")[-1]
                assert len(answer) < 300, "Answer is too long"
                outputs.append(
                    {
                        "context": sampled_context.page_content,
                        "question": question,
                        "answer": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )
            else:
                # 处理中文的逻辑
                pass
        except Exception as e:
            continue
    return outputs

# 设置批判智能体
def critique_prompt(language):
    question_groundedness_critique_prompt = None
    question_relevance_critique_prompt = None
    question_standalone_critique_prompt = None
    if language == 'en':
        # 1. 具体性（Groundedness）智能体：问题是否可以从给定的上下文中得到回答？
        question_groundedness_critique_prompt = """
                                                You will be given a context and a question.
                                                Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
                                                Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

                                                Provide your answer as follows:

                                                Answer:::
                                                Evaluation: (your rationale for the rating, as a text)
                                                Total rating: (your rating, as a number between 1 and 5)

                                                You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

                                                Now here are the question and context.

                                                Question: {question}\n
                                                Context: {context}\n
                                                Answer::: """
        # 2. 相关性（Relevance）：问题对用户是否相关？
        question_relevance_critique_prompt = """
                                                You will be given a question.
                                                Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
                                                Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

                                                Provide your answer as follows:

                                                Answer:::
                                                Evaluation: (your rationale for the rating, as a text)
                                                Total rating: (your rating, as a number between 1 and 5)

                                                You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

                                                Now here is the question.

                                                Question: {question}\n
                                                Answer::: """

        # 3. 独立（Stand-alone）：对于一个具有领域知识/互联网访问权限的人来说，问题在没有任何上下文的情况下是否可以理解？
        question_standalone_critique_prompt = """
                                                You will be given a question.
                                                Your task is to provide a 'total rating' representing how context-independant this question is.
                                                Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
                                                For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
                                                The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

                                                For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independant from the context.

                                                Provide your answer as follows:

                                                Answer:::
                                                Evaluation: (your rationale for the rating, as a text)
                                                Total rating: (your rating, as a number between 1 and 5)

                                                You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

                                                Now here is the question.

                                                Question: {question}\n
                                                Answer::: """
    elif language == 'zh':
        # 1. 事实性（Groundedness）：问题是否可基于给定上下文明确回答？
        question_groundedness_critique_prompt = """
                                                你将获得一个上下文和一个问题。
                                                你的任务是对该问题能否基于给定上下文明确回答进行'总分'评估。
                                                评分范围为1-5分：
                                                1分表示问题完全无法通过上下文回答，
                                                5分表示问题可以清晰、明确地通过上下文回答。

                                                请按以下格式提供答案：

                                                回答:::
                                                评估：（您的评分依据说明）
                                                总分：（1-5之间的评分数字）

                                                您必须在回答中提供'评估：'和'总分：'的具体内容。

                                                以下是问题和上下文：

                                                问题：{question}
                                                上下文：{context}
                                                回答::: """

        # 2. 相关性（Relevance）：问题对目标用户是否相关？
        question_relevance_critique_prompt = """
                                                你将获得一个问题。
                                                你的任务是对该问题对机器学习开发者（使用Hugging Face生态构建NLP应用）的有用性进行'总分'评估。
                                                评分范围为1-5分：
                                                1分表示问题完全无用，
                                                5分表示问题极其有用。

                                                请按以下格式提供答案：

                                                回答:::
                                                评估：（您的评分依据说明）
                                                总分：（1-5之间的评分数字）

                                                您必须在回答中提供'评估：'和'总分：'的具体内容。

                                                以下是问题：

                                                问题：{question}
                                                回答::: """

        # 3. 独立性（Stand-alone）：问题是否无需上下文即可理解？
        question_standalone_critique_prompt = """
                                                你将获得一个问题。
                                                你的任务是对该问题的独立性进行'总分'评估：
                                                1分表示问题需要额外上下文才能理解，
                                                5分表示问题本身具备完整语义（即使包含专业术语）。
                                                注：若问题包含上下文依赖表述（如"在上下文中"或"在文档中"），必须评1分。
                                                允许包含技术术语（如Gradio/Hub/Hugging Face），只要专业人员能通过文档理解即可。

                                                示例：
                                                "ViT模型导入的检查点名称是什么？"应评1分（隐含上下文依赖）
                                                "解释Hugging Face Hub的模型部署流程"可评5分（独立完整）

                                                请按以下格式提供答案：

                                                回答:::
                                                评估：（您的评分依据说明）
                                                总分：（1-5之间的评分数字）

                                                您必须在回答中提供'评估：'和'总分：'的具体内容。

                                                以下是问题：

                                                问题：{question}
                                                回答::: """
    else:
        raise ValueError("critique_prompt failed")
    return question_groundedness_critique_prompt, question_relevance_critique_prompt, question_standalone_critique_prompt

def Generating_RAG_Dataset():

    # ====================== Generating RAG Test Dataset ================================
    print("Generating critique for each QA couple...")
    language = 'en'
    llm_model = 'deepseek-r1:1.5b'
    llm_api = 'deepseek-chat'
    save_dir = "/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/eval"
    os.makedirs(save_dir, exist_ok=True)
    docs_processed = dataset_prepare()
    template = template_en_or_zh(language)
    outputs = Generate_QA_couple(llm_api, call_llm_API, docs_processed, language, template)
    question_groundedness_critique_prompt, question_relevance_critique_prompt, question_standalone_critique_prompt = critique_prompt(language)


    for output in tqdm(outputs):
        evaluations = {
            "groundedness": call_llm_API(llm_api, question_groundedness_critique_prompt.format(context=output["context"], question=output["question"]),),
            "relevance": call_llm_API(llm_api,question_relevance_critique_prompt.format(question=output["question"]),),
            "standalone": call_llm_API(llm_api,question_standalone_critique_prompt.format(question=output["question"]),),
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
        except Exception as e:
            continue

    generated_questions = pd.DataFrame.from_dict(outputs)

    # print("Evaluation dataset before filtering:")
    # generated_questions = generated_questions.loc[
    #     (generated_questions["groundedness_score"] >= 4)
    #     & (generated_questions["relevance_score"] >= 4)
    #     & (generated_questions["standalone_score"] >= 4)
    # ]
    print("============================================")
    print("Final evaluation dataset:")
    print(generated_questions)

    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)

    # 保存为 CSV（处理多行文本和引号）
    generated_questions.to_csv(
        "output/filtered_questions.csv",
        index=False,                  # 不保存索引
        quoting=csv.QUOTE_ALL,       # 强制所有字段用双引号包裹
        escapechar="\\",             # 转义特殊符号（如代码块中的引号）
        encoding="utf-8-sig"         # 兼容 Excel 的 UTF-8 BOM 格式
    )

    # 验证保存结果
    try:
        validation_df = pd.read_csv(
            "output/filtered_questions.csv",
            quoting=csv.QUOTE_ALL,
            escapechar="\\"
        )
        print("\n验证 CSV 第一行:")
        print(validation_df.iloc[0].to_dict())
    except FileNotFoundError:
        print("保存失败：文件未生成")

    

    # ====================== Generating RAG Test Dataset ================================

Generating_RAG_Dataset()