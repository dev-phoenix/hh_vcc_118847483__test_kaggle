# lib_llm__langch
# langchain

from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain.document_loaders import Docx2txtLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import HuggingFaceHub
from langchain_community.document_loaders import Docx2txtLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

import langchain_community.document_loaders as dl

# langchain_community.document_loaders
# print(dir(langchain.vectorstores))
# print(dir(dl))


# MODEL_NAME='lmsys/vicuna-13b-v1.5-16k'
MODEL_NAME='lmsys/vicuna-13b-v1.5-16k'
MODEL_NAME='lmsys/vicuna-13b-v1.5-16k'
MODEL_NAME='Defetya/qwen-4B-saiga'
MODEL_NAME='sentence-transformers/distiluse-base-multilingual-cased'
repo_id='lmsys/vicuna-13b-v1.5-16k'

def test_llm__langch():
    

    docs = []
    # Загружаем документ
    # loader = Docx2txtLoader('user_guide.docx')
    # docs = loader.load()


    print('='*30, 'RecursiveCharacterTextSplitter ...')
    # Разбиваем документ на части
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print('='*30, 'HuggingFaceEmbeddings ...')
    # Переводим части документа в эмбединги и помещаем в хранилище
    embedding = HuggingFaceEmbeddings(
        model_name='sentence-transformers/distiluse-base-multilingual-cased',
        # model_kwargs={'device':'cuda:0'},
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    retriever=[]
    print('='*30, 'Chroma ...')
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    retriever = vectorstore.as_retriever()

    print('='*30, 'HuggingFaceHub ...')
    # Подгружаем LLM с HF
    # llm = HuggingFaceHub(repo_id='lmsys/vicuna-13b-v1.5-16k')
    llm = HuggingFaceHub(repo_id='lmsys/vicuna-13b-v1.5-16k')

    # Формируем шаблон запроса к llm
    template = """Используй следующие фрагменты контекста, чтобы в конце ответить на вопрос.
    Если ты не нашел ответа, просто скажи, что не знаешь ответа. Не пытайся выдумывать ответ.
    Используй максимум три предложения и старайся отвечать максимально кратко.
    {context}
    Вопрос: {question}
    Полезный ответ: """
    print('='*30, 'PromptTemplate ...')
    prompt = PromptTemplate.from_template(template)
    print('='*30, 'prompt ...')
    print(prompt)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Формируем конвеер
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Задаем вопрос по документу
    out = rag_chain.invoke("Какое назначение у ЛИС?")

    print(out)