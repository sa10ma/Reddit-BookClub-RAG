import requests
from bs4 import BeautifulSoup
import json
import os

def scrape_sparknotes():
    documents = []
    doc_counter = 1
    
    # Loop through sections 1 to 10 (covering prologue, chapters, etc.)
    for i in range(1, 11):
        url = f"https://www.sparknotes.com/lit/bluesteye/section{i}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        print(f"Fetching Section {i}: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Skipping Section {i} (Status: {response.status_code})")
            continue
            
        soup = BeautifulSoup(response.text, 'html.parser')
        container = soup.select_one('div.mainTextContent.main-container')
        
        if not container:
            print(f"Could not find main content container for Section {i}")
            continue
            
        current_type = "summary" # Default fallback type
        
        # Find headers and paragraphs in chronological order inside the container
        elements = container.find_all(['h2', 'h3', 'p'])
        
        for el in elements:
            tag_name = el.name
            text = el.get_text().strip()
            
            # Check if the element is a section header to switch between summary/analysis
            if tag_name in ['h2', 'h3']:
                header_text = text.lower()
                if 'summary' in header_text:
                    current_type = 'summary'
                elif 'analysis' in header_text:
                    current_type = 'analysis'
                    
            # If it's a paragraph, tag it with the current chapter and type
            elif tag_name == 'p':
                if i == 1:
                    if len(text) > 60: # Filter out short boilerplate sentences
                        documents.append({
                            "id": f"doc_{doc_counter}",
                            "chapter": "Prologue",
                            "type": current_type, # 'summary' or 'analysis'
                            "text": text
                        })
                else:
                    if len(text) > 60: # Filter out short boilerplate sentences
                        documents.append({
                            "id": f"doc_{doc_counter}",
                            "chapter": f"Chapter {i-1}",
                            "type": current_type, # 'summary' or 'analysis'
                            "text": text
                        })      
                
                doc_counter += 1

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to JSON
    output_path = "data/sparknotes_bluesteye.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
        
    print(f"\nSuccessfully scraped {len(documents)} total paragraphs across 10 sections!")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    scrape_sparknotes()