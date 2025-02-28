import asyncio
import os
import re
from pathlib import Path
from typing import List
from urllib.parse import urljoin
from fake_useragent import UserAgent
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from src.aux_utils.logging_utils import setup_logger

doc_logger = setup_logger(__name__, "doc_scrapper.log")

class DocScrapperAgent:
    def __init__(self, output_dir: str = "modules_doc"):
        self.output_dir = output_dir
        
    async def get_doc_links(self, url: str) -> List[str]:
        """Extracts internal documentation links with context-aware filtering"""
        config = CrawlerRunConfig(
            exclude_external_links=True,
            process_iframes=True,
            remove_overlay_elements=True,
            word_count_threshold=10,
            excluded_tags=['nav', 'footer', 'header'],
            exclude_social_media_links=True,
            css_selector="main.doc-contents, article, section",
            cache_mode="bypass"
        )
        
        ua = UserAgent()
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=True,
            java_script_enabled=True,
            text_mode=True,
            user_agent=ua.random
        )
        
        async with AsyncWebCrawler(verbose=True, headless=True, config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=config)
            
            if not result.success:
                doc_logger.error(f"Error fetching links: {result.error_message}")
                return []

            base_url = result.url if result.url else url
            internal_links = [urljoin(base_url, link['href'])
                          for link in result.links.get('internal', [])
                          if link.get('href')]
            
            return list(set(internal_links))

    def _clean_content(self, content: str) -> str:
        """Clean and format the scraped content"""
        # Remove URLs
        content = re.sub(r"https?://\S+", "", content)
        # Remove empty lines
        content = "\n".join(line for line in content.split("\n") if line.strip())
        return content.strip()

    def _get_safe_filename(self, url: str) -> str:
        """Convert URL to safe filename"""
        name = url.split("/")[-1]
        if not name:
            name = url.split("/")[-2]
        name = re.sub(r'[^\w\-_.]', '_', name)
        return f"{name}.md"

    async def fetch_full_doc(self, doc_url: str, timeout: int = 30) -> None:
        """Fetch and save all documentation pages as markdown files"""
        doc_logger.info(f"Starting documentation fetch for: {doc_url}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get all documentation subpages
        sub_urls = await self.get_doc_links(doc_url)
        doc_logger.info(f"Found {len(sub_urls)} documentation pages")
        
        for url in sub_urls:
            try:
                ua = UserAgent()
                browser_config = BrowserConfig(
                    browser_type="chromium",
                    headless=True,
                    verbose=True,
                    text_mode=True,
                    user_agent=ua.random
                )
                
                async with AsyncWebCrawler(verbose=True, headless=True, config=browser_config) as crawler:
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, cache_mode=CacheMode.BYPASS),
                        timeout=timeout
                    )
                    
                    if result and result.markdown_v2:
                        content = self._clean_content(result.markdown_v2.raw_markdown)
                        
                        # Save to markdown file
                        filename = self._get_safe_filename(url)
                        filepath = Path(self.output_dir) / filename
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"# {url}\n\n{content}")
                        
                        doc_logger.info(f"Saved documentation page: {filename}")
                    
            except Exception as e:
                doc_logger.error(f"Error processing {url}: {str(e)}")

async def main():
    doc_url = input("Enter documentation URL: ").strip()
    print(f"\nAnalyzing documentation structure for: {doc_url}")
    
    try:
        agent = DocScrapperAgent()
        await agent.fetch_full_doc(doc_url)
        print(f"\nDocumentation has been saved to the {agent.output_dir} directory")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
