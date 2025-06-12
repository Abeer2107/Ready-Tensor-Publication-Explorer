import os
from dotenv import load_dotenv
from typing import Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
import google.generativeai as genai

load_dotenv()

class GeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    model: Optional[Any] = None

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__()
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            raise ValueError(f"Invalid Gemini model config: {str(e)}")

    def _call(
            self,
            prompt: str,
            stop: Optional[list] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise

    @property
    def _llm_type(self) -> str:
        return "gemini"

class RAGPipeline:
    def __init__(self):
        # Validate API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        try:
            genai.configure(api_key=api_key)
            genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            raise ValueError(f"Invalid API key: {str(e)}")

        # Initialize LLM
        self.llm = GeminiLLM(model_name="gemini-1.5-flash")

        # Load prompt template
        with open("src/prompt.txt", "r", encoding="utf-8") as file:
            prompt_text = file.read()

        self.prompt_template = PromptTemplate(input_variables=["context", "question"], template=prompt_text)
        self.chain = None

    def create_chain(self, retriever):
        try:
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            return self.chain
        except Exception as e:
            raise