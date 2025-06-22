import requests
import pandas as pd
import json
import time
import os
from typing import List, Dict, Optional
import random

class OpenAlexFetcher:
    def __init__(self, email: Optional[str] = None):
        """
        Initialize OpenAlex fetcher
        email: Your email for polite API usage (recommended but not required)
        """
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.session = requests.Session()
        
        # Enhanced headers for better API compatibility
        headers = {
            'User-Agent': f'OpenAlexFetcher/1.0 (mailto:{email})' if email else 'OpenAlexFetcher/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        self.session.headers.update(headers)
        
        # Create cache directory
        self.cache_dir = "data/openalex_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def search_papers(self, 
                     query: str, 
                     limit: int = 5000,
                     fields: Optional[List[str]] = None,
                     filters: Optional[Dict[str, str]] = None,
                     sort: str = "cited_by_count:desc") -> pd.DataFrame:
        """
        Search for papers using OpenAlex API
        
        Args:
            query: Search query
            limit: Number of papers to fetch (can be very large)
            fields: List of fields to retrieve
            filters: Additional filters (e.g., {"publication_year": "2020-2024"})
            sort: How to sort results
        """
        
        if fields is None:
            fields = [
                "id", "title", "abstract", "publication_year", "cited_by_count",
                "authorships", "concepts", "referenced_works", "related_works",
                "open_access", "doi", "pdf_url", "landing_page_url"
            ]
        
        cache_file = os.path.join(self.cache_dir, f"search_{query.replace(' ', '_')[:50]}_{limit}.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"Loaded {len(cached_data)} papers from cache")
                return pd.DataFrame(cached_data)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Cache error: {e}, fetching fresh data")
        
        papers = []
        per_page = 200  # OpenAlex max per request
        cursor = "*"
        
        print(f"Searching OpenAlex for: '{query}' (target: {limit} papers)")
        
        while len(papers) < limit and cursor:
            # Build parameters
            params = {
                "search": query,
                "per-page": min(per_page, limit - len(papers)),
                "cursor": cursor,
                "select": ",".join(fields),
                "sort": sort
            }
            
            # Add filters
            if filters:
                filter_str = ",".join([f"{k}:{v}" for k, v in filters.items()])
                params["filter"] = filter_str
            
            try:
                print(f"Making request to: {self.base_url}/works")
                print(f"Headers: {dict(self.session.headers)}")
                print(f"Params: {params}")
                
                response = self.session.get(f"{self.base_url}/works", params=params, timeout=30)
                
                print(f"Response status: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 403:
                    print("403 Forbidden - Checking response body for details:")
                    try:
                        error_body = response.text[:500]  # First 500 chars
                        print(f"Error response: {error_body}")
                    except:
                        pass
                    
                    # Try with minimal parameters
                    minimal_params = {
                        "search": query,
                        "per-page": min(25, limit - len(papers))  # Smaller batch maximum 200 papers per call
                    }
                    print("Retrying with minimal parameters...")
                    response = self.session.get(f"{self.base_url}/works", params=minimal_params, timeout=30)
                    print(f"Retry response status: {response.status_code}")
                
                response.raise_for_status()
                data = response.json()
                
                batch_papers = data.get("results", [])
                papers.extend(batch_papers)
                
                # Get next cursor
                cursor = data.get("meta", {}).get("next_cursor")
                
                print(f"Fetched {len(papers)}/{limit} papers...")
                
                # ensure not overloading the API
                time.sleep(0.5)
                
            except requests.exceptions.Timeout:
                print("Request timed out - trying again with longer timeout")
                time.sleep(2)
                continue
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                print(f"Response status code: {getattr(e.response, 'status_code', 'N/A')}")
                print(f"Response text: {getattr(e.response, 'text', 'N/A')[:200]}")
                
                if len(papers) > 0:
                    print(f"Continuing with {len(papers)} papers fetched so far")
                    break
                else:
                    # Try a simple test request
                    print("Attempting simple test request...")
                    try:
                        test_response = self.session.get(f"{self.base_url}/works", 
                                                       params={"search": "test", "per-page": 1}, 
                                                       timeout=15)
                        print(f"Test request status: {test_response.status_code}")
                        if test_response.status_code == 200:
                            print("Basic API access works - issue might be with specific parameters")
                    except Exception as test_e:
                        print(f"Test request also failed: {test_e}")
                    
                    raise e
        
        # Process and clean the data
        processed_papers = []
        for paper in papers:
            processed_paper = self._process_paper(paper)
            processed_papers.append(processed_paper)
        
        # Cache results for local use
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(processed_papers, f, indent=2, ensure_ascii=False)
        except Exception as cache_e:
            print(f"Warning: Could not cache results: {cache_e}")
        
        print(f"Successfully fetched {len(processed_papers)} papers")
        
        df = pd.DataFrame(processed_papers)
        return df
    
    def _convert_inverted_index_to_text(self, inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """Convert OpenAlex inverted index format to readable text"""
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        
        try:
            # Create a list to hold words at their positions
            word_positions = {}
            
            # Place each word at its positions
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions[pos] = word
            
            # Sort by position and join
            if not word_positions:
                return ""
            
            max_pos = max(word_positions.keys())
            text_parts = []
            
            for i in range(max_pos + 1):
                if i in word_positions:
                    text_parts.append(word_positions[i])
                else:
                    text_parts.append("")  # Handle missing positions
            
            # Join and clean up extra spaces
            text = " ".join(text_parts)
            # Remove multiple spaces
            import re
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            print(f"Error converting inverted index: {e}")
            return ""
    
    def _process_paper(self, paper: Dict) -> Dict:
        # Extract authors
        authors = []
        if paper.get("authorships"):
            for auth in paper["authorships"]:
                author_info = auth.get("author", {})
                if author_info and author_info.get("display_name"):
                    authors.append({
                        "name": author_info["display_name"],
                        "id": author_info.get("id", ""),
                        "institution": auth.get("institutions", [{}])[0].get("display_name", "") if auth.get("institutions") else ""
                    })
        
        # Extract concepts/topics
        concepts = []
        if paper.get("concepts"):
            concepts = [
                {
                    "name": concept["display_name"],
                    "level": concept.get("level", 0),
                    "score": concept.get("score", 0)
                }
                for concept in paper["concepts"][:10]  # Top 10 concepts
            ]
        
        # Extract URLs
        pdf_url = None
        if paper.get("open_access", {}).get("oa_url"):
            pdf_url = paper["open_access"]["oa_url"]
        
        # Convert inverted index to readable abstract
        abstract_text = self._convert_inverted_index_to_text(paper.get("abstract_inverted_index"))
        
        # Extract referenced works
        referenced_works = paper.get("referenced_works", [])
        
        # Extract related works
        related_works = paper.get("related_works", [])
        
        return {
            "openalex_id": paper.get("id", ""),
            "title": paper.get("title", ""),
            "abstract": abstract_text,
            "publication_year": paper.get("publication_year"),
            "cited_by_count": paper.get("cited_by_count", 0),
            "authors": authors,
            "concepts": concepts,
            "doi": paper.get("doi", ""),
            "pdf_url": pdf_url,
            "landing_page_url": paper.get("landing_page_url", ""),
            "is_open_access": paper.get("open_access", {}).get("is_oa", False),
            "referenced_works": referenced_works,
            "referenced_works_count": len(referenced_works),
            "related_works": related_works,
            "type": paper.get("type", ""),
            "venue": paper.get("primary_location", {}).get("source", {}).get("display_name", "") if paper.get("primary_location") else ""
        }
    
    def get_referenced_work_ids(self, referenced_works: List[str]) -> List[str]:
        """
        Extract OpenAlex work IDs from referenced works URLs
        
        Args:
            referenced_works: List of OpenAlex URLs
            
        Returns:
            List of work IDs (e.g., ['W1678356000', 'W1949087994', ...])
        """
        work_ids = []
        for url in referenced_works:
            if url and url.startswith("https://openalex.org/W"):
                work_id = url.split("/")[-1]  # Extract the work ID
                work_ids.append(work_id)
        return work_ids
    
    def fetch_referenced_works_details(self, referenced_works: List[str], 
                                     fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch details for referenced works
        
        Args:
            referenced_works: List of OpenAlex work URLs
            fields: Fields to retrieve for each referenced work
            
        Returns:
            DataFrame with details of referenced works
        """
        if not referenced_works:
            return pd.DataFrame()
        
        if fields is None:
            fields = ["id", "title", "publication_year", "cited_by_count", "authorships"]
        
        # Extract work IDs from URLs
        work_ids = self.get_referenced_work_ids(referenced_works)
        
        if not work_ids:
            return pd.DataFrame()
        
        # Batch fetch referenced works (OpenAlex allows filtering by multiple IDs)
        referenced_papers = []
        batch_size = 50  # Process in batches to avoid URL length limits
        
        for i in range(0, len(work_ids), batch_size):
            batch_ids = work_ids[i:i + batch_size]
            ids_filter = "|".join(batch_ids)
            
            params = {
                "filter": f"openalex_id:{ids_filter}",
                "select": ",".join(fields),
                "per-page": len(batch_ids)
            }
            
            try:
                response = self.session.get(f"{self.base_url}/works", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                batch_papers = data.get("results", [])
                referenced_papers.extend(batch_papers)
                
                time.sleep(0.5)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching referenced works batch: {e}")
                continue
        
        # Process the referenced papers
        processed_papers = []
        for paper in referenced_papers:
            processed_paper = self._process_paper(paper)
            processed_papers.append(processed_paper)
        
        return pd.DataFrame(processed_papers)
    
    def test_connection(self) -> bool:
        """Test basic connectivity to OpenAlex API"""
        try:
            response = self.session.get(f"{self.base_url}/works", 
                                      params={"search": "test", "per-page": 1}, 
                                      timeout=10)
            print(f"Test connection status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"Test connection failed: {e}")
            return False