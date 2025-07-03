from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
import pymongo
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated
from private_gpt.settings.settings import settings, Settings

ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])
# CONNECTION_STRING = "mongodb+srv://user:pass@cluster.mongodb.net/myFirstDatabase"
# mongo_client = pymongo.MongoClient(CONNECTION_STRING)

class IngestTextBody(BaseModel):
    file_name: str = Field(examples=["Avatar: The Last Airbender"])
    text: str = Field(
        examples=[
            "Avatar is set in an Asian and Arctic-inspired world in which some "
            "people can telekinetically manipulate one of the four elements—water, "
            "earth, fire or air—through practices known as 'bending', inspired by "
            "Chinese martial arts."
        ]
    )


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]

class IngestFileParams(BaseModel):
    collection: str = Field(examples=["my_collection"])

@ingest_router.post("/ingest", tags=["Ingestion"], deprecated=True)
def ingest(request: Request, file: UploadFile) -> IngestResponse:
    """Ingests and processes a file.

    Deprecated. Use ingest/file instead.
    """
    return ingest_file(request, file)


@ingest_router.post("/ingest/file", tags=["Ingestion"])
def ingest_file(request: Request, file: UploadFile, params:IngestFileParams=Depends()) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    Most common document
    formats are supported, but you may be prompted to install an extra dependency to
    manage a specific file type.

    A file can generate different Documents (for example a PDF generates one Document
    per page). All Documents IDs are returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). Those IDs
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    custom_settings = settings()
    custom_settings.vectorstore.collection = params.collection
    
    # Update the settings in the injector
    request.state.injector.binder.bind(Settings, custom_settings)
    service = request.state.injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_bin_data(file.filename, file.file)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.post("/ingest/text", tags=["Ingestion"])
def ingest_text(request: Request, body: IngestTextBody) -> IngestResponse:
    """Ingests and processes a text, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    A Document will be generated with the given text. The Document
    ID is returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). That ID
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    if len(body.file_name) == 0:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_text(body.file_name, body.text)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)

@ingest_router.post("/ingest/mongo", tags=["Ingestion"])
def ingest_mongo(request: Request, body: IngestTextBody) -> IngestResponse:
    """Ingests and processes a text, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    A Document will be generated with the given text. The Document
    ID is returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). That ID
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    # Get MongoDB connection settings from request params
    custom_settings = settings()
    custom_settings.vectorstore.collection = body.collection if hasattr(body, 'collection') else 'default'
    
    # Update the settings in the injector 
    request.state.injector.binder.bind(Settings, custom_settings)

    try:
        # Connect to MongoDB
        db = mongo_client[body.database]
        collection = db[body.collection]

        # Fetch all documents from collection
        documents = collection.find({})
        
        all_ingested_docs = []
        
        # Iterate through documents and ingest each one
        for doc in documents:
            # Convert MongoDB document to string representation
            doc_str = str(doc)
            
            # Generate a filename using document ID
            file_name = f"mongo_doc_{str(doc['_id'])}.txt"
            
            # Get service instance
            service = request.state.injector.get(IngestService)
            
            # Ingest the document text
            ingested_docs = service.ingest_text(file_name, doc_str)
            all_ingested_docs.extend(ingested_docs)

        mongo_client.close()
        return IngestResponse(object="list", model="private-gpt", data=all_ingested_docs)

    except Exception as e:
        raise HTTPException(500, f"Error ingesting MongoDB documents: {str(e)}")
    # service = request.state.injector.get(IngestService)
    # if len(body.file_name) == 0:
    #     raise HTTPException(400, "No file name provided")
    

    ingested_documents = service.ingest_text(body.file_name, body.text)

    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested(request: Request) -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    ingested_documents = service.list_ingested()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(request: Request, doc_id: str) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = request.state.injector.get(IngestService)
    service.delete(doc_id)
