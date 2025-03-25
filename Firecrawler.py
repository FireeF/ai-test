"""Firecrawl Web Reader."""
from typing import List, Optional, Dict, Callable
from pydantic import Field
import datetime

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

def cirilica_u_latinicu(text):
  """
  Funkcija koja preslovljava ćirilična slova unutar teksta na latinicu.

  Argumenti:
    tekst: String koji se preslovljava.

  Vraća:
    String s preslovljenim ćiriličnim slovima na latinicu.
  """

  prevodi_cirilica_latinica = {
      "а": "a",
      "б": "b",
      "в": "v",
      "г": "g",
      "д": "d",
      "ђ": "dj",
      "е": "e",
      "ж": "z",
      "з": "z",
      "и": "i",
      "ј": "j",
      "к": "k",
      "л": "l",
      "љ": "lj",
      "м": "m",
      "н": "n",
      "њ": "nj",
      "о": "o",
      "п": "p",
      "р": "r",
      "с": "s",
      "т": "t",
      "ћ": "ć",
      "у": "u",
      "ф": "f",
      "х": "h",
      "ц": "c",
      "ч": "č",
      "џ": "dž",
      "ш": "š",
      "А": "A",
      "Б": "B",
      "В": "V",
      "Г": "G",
      "Д": "D",
      "Ђ": "Dj",
      "Е": "E",
      "Ж": "Z",
      "З": "Z",
      "И": "I",
      "Ј": "J",
      "К": "K",
      "Л": "L",
      "Љ": "Lj",
      "М": "M",
      "Н": "N",
      "Њ": "Nj",
      "О": "O",
      "П": "P",
      "Р": "R",
      "С": "S",
      "Т": "T",
      "Ћ": "Ć",
      "У": "U",
      "Ф": "F",
      "Х": "H",
      "Ц": "C",
      "Ч": "Č",
      "Џ": "Dž",
      "Ш": "Š",
  }

  new_text = ""
  for letter in text:
    if letter in prevodi_cirilica_latinica:
      new_text += prevodi_cirilica_latinica[letter]
    else:
      new_text += letter

  return new_text

class FireCrawlWebReader(BasePydanticReader):
    """turn a url to llm accessible markdown with `Firecrawl.dev`.

    Args:
    api_key: The Firecrawl API key.
    api_url: url to be passed to FirecrawlApp for local deployment
    url: The url to be crawled (or)
    mode: The mode to run the loader in. Default is "crawl".
    Options include "scrape" (single url),
    "crawl" (all accessible sub pages),
    "search" (search for content), and
    "extract" (extract structured data from URLs using a prompt).
    params: The parameters to pass to the Firecrawl API.
    Examples include crawlerOptions.
    For more details, visit: https://docs.firecrawl.dev/sdks/python

    """

    firecrawl: Optional[object] = Field(None)
    api_key: str
    api_url: Optional[str]
    mode: Optional[str]
    params: Optional[dict]

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        mode: Optional[str] = "scrape",
        params: Optional[dict] = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(api_key=api_key, api_url=api_url, mode=mode, params=params)
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )
        if api_url:
            self.firecrawl = FirecrawlApp(api_key=api_key, api_url=api_url)
        else:
            self.firecrawl = FirecrawlApp(api_key=api_key)

    def _filter_metadata(self, metadata: Dict) -> Dict:
        """Filter metadata to only keep url and title, and add timestamp.
        
        Args:
            metadata: The original metadata dictionary
            
        Returns:
            Dict: Filtered metadata with only url, title and timestamp
        """
        filtered_metadata = {}
        
        # Only extract the exact keys "url" and "title"
        if "url" in metadata:
            filtered_metadata["url"] = metadata["url"]
            
        if "title" in metadata:
            filtered_metadata["title"] = cirilica_u_latinicu(metadata["title"])
        
        # Add timestamp
        filtered_metadata['timestamp'] = datetime.datetime.now().isoformat()
        
        return filtered_metadata

    @classmethod
    def class_name(cls) -> str:
        return "Firecrawl_reader"

    def load_data(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        urls: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            url (Optional[str]): URL to scrape or crawl.
            query (Optional[str]): Query to search for.
            urls (Optional[List[str]]): List of URLs for extract mode.

        Returns:
            List[Document]: List of documents.

        Raises:
            ValueError: If invalid combination of parameters is provided.
        """
        if sum(x is not None for x in [url, query, urls]) != 1:
            raise ValueError("Exactly one of url, query, or urls must be provided.")

        documents = []

        if self.mode == "scrape":
            # [SCRAPE] params: https://docs.firecrawl.dev/api-reference/endpoint/scrape
            if url is None and urls is None:
                raise ValueError("URL or URLS must be provided for scrape mode.")
            elif url is not None and urls is not None:
                raise ValueError("Only one of URL or URLS should be provided, not both.")
            
            if url:
                firecrawl_docs = self.firecrawl.scrape_url(url, params=self.params)
                documents.append(
                    Document(
                        text=firecrawl_docs.get("markdown", ""),
                        metadata=self._filter_metadata(firecrawl_docs.get("metadata", {})),
                    )
                )

            elif urls:
                firecrawl_docs = self.firecrawl.batch_scrape_urls(urls, params=self.params)
                #print(firecrawl_docs)
                # Extract the data array from the response
                if isinstance(firecrawl_docs, dict) and 'data' in firecrawl_docs:
                    docs_list = firecrawl_docs.get('data', [])
                    for doc in docs_list:

                        text = cirilica_u_latinicu(doc.get("markdown", ""))

                        documents.append(
                            Document(
                                text=text,
                                metadata=self._filter_metadata(doc.get("metadata", {})),
                            )
                        )
                else:
                    print(f"Unexpected response format from batch_scrape_urls: {type(firecrawl_docs)}")

        elif self.mode == "crawl":
            # [CRAWL] params: https://docs.firecrawl.dev/api-reference/endpoint/crawl-post
            if url is None:
                raise ValueError("URL must be provided for crawl mode.")
            firecrawl_docs = self.firecrawl.crawl_url(url, params=self.params)
            firecrawl_docs = firecrawl_docs.get("data", [])
            for doc in firecrawl_docs:
                documents.append(
                    Document(
                        text=doc.get("markdown", ""),
                        metadata=self._filter_metadata(doc.get("metadata", {})),
                    )
                )
        elif self.mode == "search":
            # [SEARCH] params: https://docs.firecrawl.dev/api-reference/endpoint/search
            if query is None:
                raise ValueError("Query must be provided for search mode.")

            # Remove query from params if it exists to avoid duplicate
            search_params = self.params.copy() if self.params else {}
            if "query" in search_params:
                del search_params["query"]

            # Get search results
            search_response = self.firecrawl.search(query, params=search_params)

            # Handle the search response format
            if isinstance(search_response, dict):
                # Check for success
                if search_response.get("success", False):
                    # Get the data array
                    search_results = search_response.get("data", [])

                    # Process each search result
                    for result in search_results:
                        # Extract text content (prefer markdown if available)
                        text = result.get("markdown", "")
                        if not text:
                            # Fall back to description if markdown is not available
                            text = result.get("description", "")

                        # Extract metadata
                        metadata = {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                        }

                        # Create document
                        documents.append(
                            Document(
                                text=text,
                                metadata=self._filter_metadata(metadata),
                            )
                        )
                else:
                    # Handle unsuccessful response
                    warning = search_response.get("warning", "Unknown error")
                    print(f"Search was unsuccessful: {warning}")
                    documents.append(
                        Document(
                            text=f"Search for '{query}' was unsuccessful: {warning}",
                            metadata=self._filter_metadata({
                                "url": "",
                                "title": f"Error searching for '{query}'",
                            }),
                        )
                    )
            else:
                # Handle unexpected response format
                print(f"Unexpected search response format: {type(search_response)}")
                documents.append(
                    Document(
                        text=str(search_response),
                        metadata=self._filter_metadata({
                            "url": "",
                            "title": f"Search for '{query}'",
                        }),
                    )
                )
        elif self.mode == "extract":
            # [EXTRACT] params: https://docs.firecrawl.dev/api-reference/endpoint/extract
            if urls is None:
                # For backward compatibility, convert single URL to list if provided
                if url is not None:
                    urls = [url]
                else:
                    raise ValueError("URLs must be provided for extract mode.")

            # Ensure we have a prompt in params
            extract_params = self.params.copy() if self.params else {}
            if "prompt" not in extract_params:
                raise ValueError("A 'prompt' parameter is required for extract mode.")

            # Prepare the payload according to the new API structure
            payload = {"prompt": extract_params.pop("prompt")}

            # Call the extract method with the urls and params
            extract_response = self.firecrawl.extract(urls=urls, params=payload)

            # Handle the extract response format
            if isinstance(extract_response, dict):
                # Check for success
                if extract_response.get("success", False):
                    # Get the data from the response
                    extract_data = extract_response.get("data", {})

                    # Get the sources if available
                    sources = extract_response.get("sources", {})

                    # Convert the extracted data to text
                    if extract_data:
                        # Convert the data to a formatted string
                        text_parts = []
                        for key, value in extract_data.items():
                            text_parts.append(f"{key}: {value}")

                        text = "\n".join(text_parts)

                        # Create metadata with just the URL
                        metadata = {
                            "url": urls[0] if urls and len(urls) > 0 else "",
                            "title": "Extracted data",
                        }

                        # Create document
                        documents.append(
                            Document(
                                text=text,
                                metadata=self._filter_metadata(metadata),
                            )
                        )
                    else:
                        # Handle empty data in successful response
                        print("Extract response successful but no data returned")
                        documents.append(
                            Document(
                                text="Extraction was successful but no data was returned",
                                metadata=self._filter_metadata({
                                    "url": urls[0] if urls and len(urls) > 0 else "",
                                    "title": "No data extracted",
                                }),
                            )
                        )
                else:
                    # Handle unsuccessful response
                    warning = extract_response.get("warning", "Unknown error")
                    print(f"Extraction was unsuccessful: {warning}")
                    documents.append(
                        Document(
                            text=f"Extraction was unsuccessful: {warning}",
                            metadata=self._filter_metadata({
                                "url": urls[0] if urls and len(urls) > 0 else "",
                                "title": "Extraction error",
                            }),
                        )
                    )
            else:
                # Handle unexpected response format
                print(f"Unexpected extract response format: {type(extract_response)}")
                documents.append(
                    Document(
                        text=str(extract_response),
                        metadata=self._filter_metadata({
                            "url": urls[0] if urls and len(urls) > 0 else "",
                            "title": "Extract response",
                        }),
                    )
                )
        else:
            raise ValueError(
                "Invalid mode. Please choose 'scrape', 'crawl', 'search', or 'extract'."
            )

        return documents
