import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_shl_catalog():
    base_url = "https://www.shl.com/products/product-catalog/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # --- PHASE 1: Scrape Listings ---
    
    assessments = []
    start_index = 0
    print("=== Phase 1: Scraping Listing Pages ===")

    while True:
        print(f"Fetching page starting at index {start_index}...")
        url = f"{base_url}?start={start_index}&type=1"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch page. Status code: {response.status_code}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        table_wrapper = soup.find('div', class_='custom__table-responsive')
        
        if not table_wrapper:
            print("Could not find the table on this page. Listing Phase complete.")
            break
            
        table = table_wrapper.find('table')
        if not table:
            break
            
        rows = table.find_all('tr')
        data_rows = rows[1:] if len(rows) > 1 else []
        items_found = False 
        
        for row in data_rows:
            title_td = row.find('td', class_='custom__table-heading__title')
            if not title_td:
                continue
                
            link_tag = title_td.find('a')
            if link_tag:
                items_found = True
                name = link_tag.text.strip()
                href = link_tag.get('href')
                full_url = href if href.startswith('http') else f"https://www.shl.com{href}"
                
                type_td = row.find('td', class_='product-catalogue__keys')
                test_type = type_td.text.strip() if type_td else ""
                
                assessments.append({
                    "name": name,
                    "url": full_url,
                    "test_type": test_type
                })
        
        if not items_found:
            print("No more items found on page. Listing Phase complete.")
            break
            
        start_index += 12
        time.sleep(1.5) 

    # --- PHASE 2: Scrape Detail Pages ---
    print("\n=== Phase 2: Scraping Detail Pages for Descriptions ===")
    total = len(assessments)
    
    for i, item in enumerate(assessments, 1):
        print(f"[{i}/{total}] Fetching details for: {item['name']}")
        try:
            resp = requests.get(item['url'], headers=headers, timeout=15)
            if resp.status_code == 200:
                detail_soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Helper function from Claude's logic
                def text_after_h4(label: str) -> str:
                    for h4 in detail_soup.find_all("h4"):
                        if label in h4.get_text(strip=True).lower():
                            sibling = h4.find_next_sibling()
                            return sibling.get_text(separator=" ", strip=True) if sibling else ""
                    return ""

                # Extract the description
                item['description'] = text_after_h4("description")
                
                # Extract job levels (great context for the LLM later!)
                raw_levels = text_after_h4("job level")
                item['job_levels'] = [l.strip() for l in raw_levels.split(",") if l.strip()]
                
        except Exception as e:
            print(f"  [Error] Failed to fetch details for {item['name']}: {e}")
            item['description'] = ""
            item['job_levels'] = []
            
        # Be polite to the server
        time.sleep(1.5)

    # Save the fully enriched data
    with open('shl_catalog_full.json', 'w', encoding='utf-8') as f:
        json.dump(assessments, f, indent=4)
        
    print(f"\nSuccessfully saved {len(assessments)} fully enriched assessments to shl_catalog_full.json")

if __name__ == "__main__":
    scrape_shl_catalog()