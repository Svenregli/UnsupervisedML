import requests
import pandas as pd
import json
import time
import os
from typing import List, Dict, Optional
import random

class OpenAlexFetcher:
    def __init__(self, email: str = None):
        """
        Initialize OpenAlex fetcher
        email: Your email for polite API usage (recommended but not required)
        """
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.session = requests.Session()
        if email:
            self.session.headers.update({'User-Agent': f'Research Script (mailto:{email})'})
        
        # Create cache directory
        self.cache_dir = "data/openalex_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def search_papers(self, 
                     query: str, 
                     limit: int = 5000,
                     fields: List[str] = None,
                     filters: Dict[str, str] = None,
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
                print(f"✅ Loaded {len(cached_data)} papers from cache")
                return pd.DataFrame(cached_data)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Cache error: {e}, fetching fresh data")
        
        papers = []
        per_page = 200  # OpenAlex max per request
        cursor = "*"
        
        print(f"🔍 Searching OpenAlex for: '{query}' (target: {limit} papers)")
        
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
                response = self.session.get(f"{self.base_url}/works", params=params)
                response.raise_for_status()
                data = response.json()
                
                batch_papers = data.get("results", [])
                papers.extend(batch_papers)
                
                # Get next cursor
                cursor = data.get("meta", {}).get("next_cursor")
                
                print(f"📄 Fetched {len(papers)}/{limit} papers...")
                
                # Be polite to the API
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error fetching data: {e}")
                if len(papers) > 0:
                    print(f"Continuing with {len(papers)} papers fetched so far")
                break
        
        # Process and clean the data
        processed_papers = []
        for paper in papers:
            processed_paper = self._process_paper(paper)
            processed_papers.append(processed_paper)
        
        # Cache results
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(processed_papers, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully fetched {len(processed_papers)} papers")
        
        df = pd.DataFrame(processed_papers)
        return df
    
    def _process_paper(self, paper: Dict) -> Dict:
        """Process raw OpenAlex paper data into cleaner format"""
        
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
        
        return {
            "openalex_id": paper.get("id", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract_inverted_index", ""),  # Note: OpenAlex uses inverted index
            "publication_year": paper.get("publication_year"),
            "cited_by_count": paper.get("cited_by_count", 0),
            "authors": authors,
            "concepts": concepts,
            "doi": paper.get("doi", ""),
            "pdf_url": pdf_url,
            "landing_page_url": paper.get("landing_page_url", ""),
            "is_open_access": paper.get("open_access", {}).get("is_oa", False),
            "referenced_works_count": len(paper.get("referenced_works", [])),
            "type": paper.get("type", ""),
            "venue": paper.get("primary_location", {}).get("source", {}).get("display_name", "") if paper.get("primary_location") else ""
        }
    
    def get_paper_references(self, openalex_id: str) -> List[Dict]:
        """Get references for a specific paper"""
        cache_file = os.path.join(self.cache_dir, f"refs_{openalex_id.split('/')[-1]}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        try:
            response = self.session.get(f"{self.base_url}/works/{openalex_id}")
            response.raise_for_status()
            paper_data = response.json()
            
            referenced_works = paper_data.get("referenced_works", [])
            
            # Fetch details for referenced works (in batches)
            references = []
            batch_size = 50
            
            for i in range(0, len(referenced_works), batch_size):
                batch_ids = referenced_works[i:i+batch_size]
                id_filter = "|".join([id.split("/")[-1] for id in batch_ids])
                
                params = {
                    "filter": f"openalex_id:{id_filter}",
                    "select": "id,title,publication_year,cited_by_count,authorships"
                }
                
                batch_response = self.session.get(f"{self.base_url}/works", params=params)
                if batch_response.status_code == 200:
                    batch_data = batch_response.json()
                    references.extend(batch_data.get("results", []))
                
                time.sleep(0.1)
            
            # Cache results
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(references, f, indent=2)
            
            return references
            
        except Exception as e:
            print(f"Error fetching references: {e}")
            return []

# Example usage functions
def search_ml_papers(fetcher: OpenAlexFetcher, limit: int = 5000) -> pd.DataFrame:
    """Search for machine learning papers"""
    return fetcher.search_papers(
        query="machine learning",
        limit=limit,
        filters={
            "publication_year": "2020-2024",
            "cited_by_count": ">5"  # Only papers with some citations
        }
    )

def search_ai_papers(fetcher: OpenAlexFetcher, limit: int = 5000) -> pd.DataFrame:
    """Search for AI papers"""
    return fetcher.search_papers(
        query="artificial intelligence",
        limit=limit,
        filters={
            "publication_year": "2021-2024",
            "type": "article"
        }
    )

# Main execution example
if __name__ == "__main__":
    # Initialize fetcher (add your email for better API behavior)
    fetcher = OpenAlexFetcher(email="your.email@university.edu")
    
    # Search for papers
    print("🚀 Starting paper search...")
    
    # Example: Get 5000 recent ML papers
    df = search_ml_papers(fetcher, limit=5000)
    
    # Save to files
    os.makedirs("data", exist_ok=True)
    df.to_json("data/openalex_papers.json", orient="records", indent=2)
    df.to_csv("data/openalex_papers.csv", index=False)
    
    print(f"📊 Dataset saved with {len(df)} papers")
    print(f"📈 Year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"🎯 Average citations: {df['cited_by_count'].mean():.1f}")