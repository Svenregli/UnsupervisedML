import azure.functions as func
import logging
import json
from openAlex_api import OpenAlexFetcher
from azure.storage.blob import BlobServiceClient
import os

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def save_json_to_blob(container_name, blob_name, json_data):
    connection_string = os.environ["AzureWebJobsStorage"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except Exception:
        pass
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(json_data.encode("utf-8"), overwrite=True)

@app.route(route="paperfetcher")
def paperfetcher(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    query = req.params.get('query')
    limit_str = req.params.get('limit', '100')
    req_body = {}
    try:
        req_body = req.get_json()
    except ValueError:
        pass

    if not query:
        query = req_body.get('query')
    if not query:
        return func.HttpResponse("Please pass a 'query' in the query string or in the request body.", status_code=400)
    
    try:
        limit = int(limit_str)
        # Limit the maximum to prevent timeouts
        if limit > 1000:
            limit = 1000
            logging.warning(f"Limiting request to {limit} papers to prevent timeout")
    except ValueError:
        return func.HttpResponse("Invalid 'limit' parameter. It must be an integer.", status_code=400)
    
    fetcher = OpenAlexFetcher(email='sven.regli2@stud.unilu.ch') 
    
    # Test connection first
    logging.info("Testing OpenAlex API connection...")
    if not fetcher.test_connection():
        return func.HttpResponse("Unable to connect to OpenAlex API", status_code=503)
    
    logging.info(f"Searching for {limit} papers with query: '{query}'")
    
    try:
        # Use simpler parameters for better compatibility
        df = fetcher.search_papers(
            query=query, 
            limit=limit,
            fields=["id", "title", "abstract_inverted_index", "publication_year", "cited_by_count", "authorships", "doi"],  # Include abstract
            sort="cited_by_count:desc"
        )
        
        if len(df) == 0:
            logging.warning("No papers found for the given query")
            return func.HttpResponse(
                json.dumps({"message": "No papers found", "query": query, "results": []}),
                mimetype="application/json",
                status_code=200
            )
        
        papers_json = df.to_json(orient="records", indent=2)

        # Save to blob storage
        blob_name = f"openalex_{query.replace(' ', '_')}.json"
        try:
            save_json_to_blob("searchresults", blob_name, papers_json)
            logging.info(f"Saved search results to blob: {blob_name}")
        except Exception as blob_err:
            logging.error(f"Failed to save to blob: {blob_err}")

        return func.HttpResponse(
            body=papers_json,
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error during paper search: {e}")
        # Return more detailed error information
        error_response = {
            "error": str(e),
            "query": query,
            "limit": limit,
            "message": "Search failed - check logs for details"
        }
        return func.HttpResponse(
            json.dumps(error_response),
            mimetype="application/json", 
            status_code=500
        )