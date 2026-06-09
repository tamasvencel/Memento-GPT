import requests
from bs4 import BeautifulSoup
import re

# Target URLs from IMSDb for Nolan's mindfuck masterpieces
movies = {
    "Memento": "https://imsdb.com/scripts/Memento.html",
    "Inception": "https://imsdb.com/scripts/Inception.html",
    "The Prestige": "https://imsdb.com/scripts/Prestige,-The.html",
    "Interstellar": "https://imsdb.com/scripts/Interstellar.html",
    "Tenet": "https://imsdb.com/scripts/Tenet.html"
}

output_file = "input.txt"
total_text = ""

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Starting the extraction process...\n")

for title, url in movies.items():
    print(f"Fetching {title}...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # IMSDb stores the raw script text inside a <td class="scrtext"> tag
        script_container = soup.find('td', class_='scrtext')

        if script_container:
            # Extract text, ignoring the surrounding HTML
            raw_text = script_container.get_text()

            # Clean up the text: replace 3 or more consecutive newlines with just 2
            # This keeps the script formatting intact but removes massive blank gaps
            cleaned_text = re.sub(r'\n{3,}', '\n\n', raw_text)
            cleaned_text = cleaned_text.strip()

            total_text += f"\n\n--- BEGIN {title.upper()} ---\n\n"
            total_text += cleaned_text
            print(f"Successfully cleaned and appended {title}.")
        else:
            print(f"Warning: Could not find the script text container for {title}.")

    except Exception as e:
        print(f"Failed to fetch {title}: {e}")

# Write everything to the final input.txt file
print(f"\nWriting the compiled dataset to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(total_text)

# Calculate and print total character count
char_count = len(total_text)
print("\n" + "="*40)
print(f"SUCCESS: The file '{output_file}' has been created.")
print(f"Total character count: {char_count:,}")
print("="*40)