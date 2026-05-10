import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_shl_catalog():
    base_url = "https://www.shl.com/products/product-catalog/"
    
    # Using a standard browser User-Agent prevents instant 403 Forbidden errors.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    assessments = []
    start_index = 0
    
    print("Starting catalog scrape...")

    while True:
        print(f"Fetching page starting at index {start_index}...")
        
        # Enforce the "Individual Test Solutions" filter (?type=1) and pagination
        url = f"{base_url}?start={start_index}&type=1"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch page. Status code: {response.status_code}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Target the specific wrapper to avoid grabbing random unrelated tables
        table_wrapper = soup.find('div', class_='custom__table-responsive')
        
        if not table_wrapper:
            print("Could not find the table on this page. Scraping complete.")
            break
            
        # 2. Find all the rows (<tr>) inside the table
        table = table_wrapper.find('table')
        if not table:
            print("Could not find the table element.")
            break
        rows = table.find_all('tr')
        
        # Skip the header row
        data_rows = rows[1:] if len(rows) > 1 else []
        
        items_found = False 
        
        for row in data_rows:
            # 3. Locate the column containing the title and link
            title_td = row.find('td', class_='custom__table-heading__title')
            
            if not title_td:
                continue
                
            link_tag = title_td.find('a')
            if link_tag:
                items_found = True
                name = link_tag.text.strip()
                
                # Format the URL properly
                href = link_tag.get('href')
                full_url = href if href.startswith('http') else f"https://www.shl.com{href}"
                
                # 4. Extract the Test Type (K or P) for the final API schema
                type_td = row.find('td', class_='product-catalogue__keys')
                test_type = type_td.text.strip() if type_td else ""
                
                assessments.append({
                    "name": name,
                    "url": full_url,
                    "test_type": test_type,
                    "description": "" # Placeholder for detailed scraping later
                })
        
        if not items_found:
            print("No more items found on page. Scraping complete.")
            break
            
        # SHL increments pages by 12
        start_index += 12
        
        # Be polite to their servers and avoid rate limits
        time.sleep(1.5) 

    # Save the structured data
    with open('shl_catalog_scraped.json', 'w', encoding='utf-8') as f:
        json.dump(assessments, f, indent=4)
        
    print(f"Successfully saved {len(assessments)} assessments to shl_catalog_scraped.json")

if __name__ == "__main__":
    scrape_shl_catalog()