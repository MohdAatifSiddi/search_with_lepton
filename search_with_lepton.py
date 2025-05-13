import concurrent.futures
import glob
import json
import os
import re
import threading
import requests
import traceback
from typing import Annotated, List, Generator, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from loguru import logger

from openai import AzureOpenAI
from fastapi import FastAPI, HTTPException

################################################################################
# Constant values for the RAG model
################################################################################

REFERENCE_COUNT = 8
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5
_default_query = "Who said 'live long and prosper'?"

_rag_query_text = """[Your existing RAG prompt here]"""

stop_words = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

_more_questions_prompt = """[Your existing questions prompt here]"""

# Configure Azure OpenAI
AZURE_OPENAI_ENDPOINT = "https://deepseek6682328837.services.ai.azure.com/models"
AZURE_API_KEY = "bOCVqqzEnzy7YKnLAOFA6PF0G47iSeoiAxRwIUbQF7va7lBAjZywJQQJ99BEAC77bzfXJ3w3AAAAACOGHY4y"
MODEL_NAME = "DeepSeek-R1"

class RAG(FastAPI):
    def __init__(self):
        super().__init__()
        self.configure_backend()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.setup_routes()

    def configure_backend(self):
        self.backend = os.getenv("BACKEND", "BING").upper()
        self.search_api_key = os.getenv(f"{self.backend}_SEARCH_API_KEY")
        self.search_function = self.get_search_function()
        
        # Initialize Azure OpenAI client
        self.llm_client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    def get_search_function(self):
        search_functions = {
            "BING": search_with_bing,
            "GOOGLE": search_with_google,
            "SERPER": search_with_serper,
            "SEARCHAPI": search_with_searchapi
        }
        return search_functions.get(self.backend, search_with_bing)

    def setup_routes(self):
        self.add_api_route("/query", self.query_function, methods=["POST"])
        self.add_api_route("/health", self.health_check, methods=["GET"])

    async def health_check(self):
        return {"status": "healthy"}

    def generate_response(self, system_prompt, user_query):
        response = self.llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=1024,
            stop=stop_words
        )
        return response.choices[0].message.content

    def get_related_questions(self, query, contexts):
        try:
            prompt = _more_questions_prompt.format(context="\n\n".join([c["snippet"] for c in contexts]))
            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.5,
                max_tokens=256
            )
            questions = response.choices[0].message.content.split("\n")
            return [q.strip() for q in questions if q.strip()]
        except Exception as e:
            logger.error(f"Error generating related questions: {str(e)}")
            return []

    async def query_function(self, query_data: dict):
        try:
            query = query_data.get("query", _default_query)
            contexts = self.search_function(query, self.search_api_key)

            system_prompt = _rag_query_text.format(
                context="\n\n".join([f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)])
            )

            answer = self.generate_response(system_prompt, query)
            related_questions = self.get_related_questions(query, contexts) if query_data.get("related_questions", True) else []

            response_data = {
                "query": query,
                "answer": answer,
                "contexts": contexts,
                "citations": [{"id": i+1, "source": c["url"]} for i, c in enumerate(contexts)],
                "related_questions": related_questions
            }

            return JSONResponse(content=response_data)

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Search functions remain the same as in original code
# [Include all the search_with_* functions here without changes]

if __name__ == "__main__":
    import uvicorn
    rag = RAG()
    uvicorn.run(rag, host="74.242.216.2", port=8888)