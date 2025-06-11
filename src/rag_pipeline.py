from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
import google.generativeai as genai
from typing import Any, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

class GeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    model: Optional[Any] = None

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__()
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"


def create_rag_chain(vectorstore):
    with open("src/prompt.txt", "r", encoding="utf-8") as file:
        prompt_text = file.read()

    llm = GeminiLLM(model_name="gemini-1.5-flash")

    prompt_template = PromptTemplate(
        input_variables=["context"],
        template=prompt_text
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return chain