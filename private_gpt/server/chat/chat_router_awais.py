from fastapi import APIRouter, Depends, Request
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.settings.settings import (
    settings,
    VectorstoreSettings, ServerSettings, 
    DataSettings, UISettings,
    LLMSettings, EmbeddingSettings,
    LlamaCPPSettings, HuggingFaceSettings,
    SagemakerSettings, OpenAISettings,
    GeminiSettings, OllamaSettings,
    AzureOpenAISettings, NodeStoreSettings,
    RagSettings, QdrantSettings, 
    PostgresSettings, ClickHouseSettings,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
    to_openai_response,
    to_openai_sse_stream,
)
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.utils.auth import authenticated

chat_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class ChatBody(BaseModel):
    messages: list[OpenAIMessage]
    use_context: bool = False
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    stream: bool = False
    collection: str = "new_collection"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a rapper. Always answer with a rap.",
                        },
                        {
                            "role": "user",
                            "content": "How do you fry an egg?",
                        },
                    ],
                    "stream": False,
                    "use_context": True,
                    "include_sources": True,
                    "context_filter": {
                        "docs_ids": ["c202d5e6-7b69-4869-81cc-dd574ee8ee11"]
                    },
                }
            ]
        }
    }


@chat_router.post(
    "/chat/completions",
    response_model=None,
    responses={200: {"model": OpenAICompletion}},
    tags=["Contextual Completions"],
    openapi_extra={
        "x-fern-streaming": {
            "stream-condition": "stream",
            "response": {"$ref": "#/components/schemas/OpenAICompletion"},
            "response-stream": {"$ref": "#/components/schemas/OpenAICompletion"},
        }
    },
)
def chat_completion(
    request: Request, body: ChatBody
) -> OpenAICompletion | StreamingResponse:
    """Given a list of messages comprising a conversation, return a response.

    Optionally include an initial `role: system` message to influence the way
    the LLM answers.

    If `use_context` is set to `true`, the model will use context coming
    from the ingested documents to create the response. The documents being used can
    be filtered using the `context_filter` and passing the document IDs to be used.
    Ingested documents IDs can be found using `/ingest/list` endpoint. If you want
    all ingested documents to be used, remove `context_filter` altogether.

    When using `'include_sources': true`, the API will return the source Chunks used
    to create the response, which come from the context provided.

    When using `'stream': true`, the API will return data chunks following [OpenAI's
    streaming model](https://platform.openai.com/docs/api-reference/chat/streaming):
    ```
    {"id":"12345","object":"completion.chunk","created":1694268190,
    "model":"private-gpt","choices":[{"index":0,"delta":{"content":"Hello"},
    "finish_reason":null}]}
    ```
    """

    # stngs = settings()

    # settings = Settings(
        # server=ServerSettings,
        # data=DataSettings,
        # ui=UISettings,
        # llm=LLMSettings,
        # embedding=EmbeddingSettings,
        # llamacpp=LlamaCPPSettings,
        # huggingface=HuggingFaceSettings,
        # sagemaker=SagemakerSettings,
        # openai=OpenAISettings,
        # gemini=GeminiSettings,
        # ollama=OllamaSettings,
        # azopenai=AzureOpenAISettings,
        # nodestore=NodeStoreSettings,
        # rag=RagSettings,
        # qdrant=QdrantSettings,
        # postgres=PostgresSettings,
        # clickhouse=None,
    #     vectorstore=VectorstoreSettings(
    #                 database="qdrant",
    #                 collection="new_collection"
    #             )
    # )
    # service = request.state.injector.get(
    #     ChatService(
    #         vector_store_component=VectorStoreComponent(
    #             settings=settings
    #         )
    #     )
    # )
    # Initialize ChatService with a custom collection name
    print(f"Fetching results from {body.collection}")
    custom_settings = settings()
    custom_settings.vectorstore.collection = body.collection
    # service = request.state.injector.get(
    #     ChatService(
    #         settings=custom_settings,
    #         vector_store_component=VectorStoreComponent(settings=custom_settings),
    #         llm_component=LLMComponent,
    #         embedding_component=EmbeddingComponent,
    #         node_store_component=NodeStoreComponent,
    #     )
    # )

    # service = request.state.injector.get(ChatService)
    service = ChatService(
        settings=custom_settings,
        llm_component=LLMComponent(settings=custom_settings),
        vector_store_component=VectorStoreComponent(settings=custom_settings),
        embedding_component=EmbeddingComponent(settings=custom_settings),
        node_store_component=NodeStoreComponent(settings=custom_settings),
    )

    all_messages = [
        ChatMessage(content=m.content, role=MessageRole(m.role)) for m in body.messages
    ]
    if body.stream:
        completion_gen = service.stream_chat(
            messages=all_messages,
            use_context=body.use_context,
            context_filter=body.context_filter,
        )
        return StreamingResponse(
            to_openai_sse_stream(
                completion_gen.response,
                completion_gen.sources if body.include_sources else None,
            ),
            media_type="text/event-stream",
        )
    else:
        completion = service.chat(
            messages=all_messages,
            use_context=body.use_context,
            context_filter=body.context_filter,
        )
        return to_openai_response(
            completion.response, completion.sources if body.include_sources else None
        )
