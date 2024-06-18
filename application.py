from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import pandas as pd

app = Flask(__name__)


# Home route to display the form
@app.route("/")
def index():
    return render_template("index.html")


# Function to find broken links (404 errors) on the page
def seo_find_404(urlSoup):
    results = []
    search_links = []
    broken_links = []

    s = requests.Session()
    s.headers["User-Agent"] = "SEO Research Assistant Program"

    for link in urlSoup.find_all("a", href=True):
        search_links.append(link.get("href"))

    broken_links_count = 0
    for search_link in search_links:
        try:
            if (
                search_link.startswith("http")
                and not search_link.startswith("mailto:")
                and "javascript:" not in search_link
                and "tel:" not in search_link
            ):
                broken_query = s.get(search_link, allow_redirects=3)

                if broken_query.status_code == 404:
                    broken_links.append(search_link)
                    broken_links_count += 1

        except requests.exceptions.ConnectionError as exc:
            results.append(f"Error: {exc}")

    results.append(f"{broken_links_count} broken links were found.")
    for broken_link in broken_links:
        results.append(f"Broken link found: {broken_link}")

    return results


# Function to check for the presence of keywords in the URL
def seo_url_keywords(keywords_list, url):
    results = []
    for keyword in keywords_list:
        if keyword.casefold() in url:
            results.append(
                f"The keyword '{keyword}' was found in your URL. That's good!"
            )
        else:
            results.append(
                f"The keyword '{keyword}' was not found in your URL. Your URL may be improved by adding keywords if you lack enough of them."
            )
    return results


# Function to get backlink data using a placeholder API call (you need to replace with actual API)
def get_all_links(url):
    try:
        # Send a request to the URL
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all the anchor tags with href attributes
            anchor_tags = soup.find_all("a", href=True)

            # Extract the domain name
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # List to store the structured link information
            links_info = []

            for tag in anchor_tags:
                link = tag.get("href")
                title = tag.get_text(strip=True) or "No Title"

                # Append the information as a dictionary
                links_info.append({"website": domain, "title": title, "link": link})

            return links_info
        else:
            print(
                f"Failed to retrieve the URL: {url}. Status code: {response.status_code}"
            )
            return []
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []


# Function to generate a backlinks report
def seo_backlinks_report():
    results = []
    try:
        with open(
            "output_files/backlinks_api_response.json", "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
            page_crawled_dates = [
                datetime.strptime(key["date"], "%Y-%m-%d").date()
                for key in data["backlinks"]
            ]
            pages_to_root_counts = [int(key["count"]) for key in data["backlinks"]]

            plt.figure(dpi=200)
            plt.plot(page_crawled_dates, pages_to_root_counts, c="blue")
            plt.title("Number of backlinks over time")
            plt.xlabel("Date")
            plt.ylabel("Number of backlinks")
            plt.grid(True)
            plt.savefig(
                f"output_files/backlinks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        results.append(f"Domain Authority: {data['domain_authority']}")
        results.append(f"Page Authority: {data['page_authority']}")
        results.append("Backlink graph saved to 'output_files/' directory.")
    except Exception as exc:
        results.append(f"Error: {exc}")
    return results


import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_stop_words(file_path):
    """
    Loads stop words from the specified file.

    Args:
    - file_path: The path to the file containing stop words, one per line.

    Returns:
    - A set of stop words.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        stop_words = {word.strip().lower() for word in file}
    return stop_words


def remove_stopwords(description, field, stopwords_path="stopwords.txt"):
    """
    Removes stop words from the provided description and field strings.

    Args:
    - description: A string containing the description.
    - field: A string containing the field.
    - stopwords_path: The path to the stop words file.

    Returns:
    - Two lists: cleaned description words and cleaned field words.
    """
    stop_words = load_stop_words(stopwords_path)

    # Ensure the inputs are strings and convert if they are not
    if not isinstance(description, str):
        print(f"Converting 'description' from {type(description)} to string.")
        description = str(description)
    if not isinstance(field, str):
        print(f"Converting 'field' from {type(field)} to string.")
        field = str(field)

    description_list = [
        word.lower() for word in description.split() if word.lower() not in stop_words
    ]
    field_list = [
        word.lower() for word in field.split() if word.lower() not in stop_words
    ]

    return description_list, field_list


def read_csv_files_in_marketers_folder(folder_path="marketers"):
    """
    Reads all CSV files in the specified folder and returns a list of DataFrames along with their filenames without the .csv extension.

    Args:
    - folder_path: The path to the folder containing the CSV files. Default is 'marketers'.

    Returns:
    - A list of tuples, each containing the filename (without .csv) and its corresponding pandas DataFrame.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in the folder '{folder_path}'.")
        return []

    dataframes_with_filenames = []

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            df = pd.read_csv(file_path)
            filename_without_extension = os.path.splitext(csv_file)[
                0
            ]  # Remove the .csv extension
            dataframes_with_filenames.append((filename_without_extension, df))
            print(f"Loaded '{filename_without_extension}' successfully.")
        except Exception as e:
            print(f"Error reading '{csv_file}': {e}")

    return dataframes_with_filenames


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def match_new_marketers(description, field, titles, stopwords_path="stopwords.txt"):
    """
    Matches the combined words from the description and field with titles in CSV files using cosine similarity.

    Args:
    - description: A string containing the description.
    - field: A string containing the field.
    - stopwords_path: The path to the stop words file.

    Returns:
    - List of tuples containing (filename, similarity_score) for the top 5 files with the most similar titles based on cosine similarity.
    """
    # Remove stopwords and combine words
    description_list, field_list = remove_stopwords(description, field, stopwords_path)
    combined_words = description_list + field_list + titles.to_list()

    combined_text = " ".join(combined_words)

    # Load CSV files from marketers folder
    dataframes_with_filenames = read_csv_files_in_marketers_folder("marketers")

    vectorizer = TfidfVectorizer()

    results = []

    for filename, df in dataframes_with_filenames:
        if "title" not in df.columns:
            print(f"No 'title' column in {filename}")
            continue

        # Convert titles to strings in case they are not
        all_titles = df["title"].astype(str).tolist()
        all_titles.append(combined_text)

        # Calculate TF-IDF and cosine similarity
        tfidf_matrix = vectorizer.fit_transform(all_titles)
        cosine_similarities = cosine_similarity(
            tfidf_matrix[-1], tfidf_matrix[:-1]
        ).flatten()

        # Calculate the average similarity for each file
        avg_similarity = cosine_similarities.mean()

        results.append(
            {
                "filename": filename.split(".")[
                    0
                ],  # Strip .csv extension from filename
                "average_similarity": avg_similarity,
            }
        )

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Sort results by similarity score in descending order
    top_5_results = results_df.sort_values(
        by="average_similarity", ascending=False
    ).head(5)

    # Extract top 5 filenames and similarity scores
    top_5_filenames = top_5_results["filename"].tolist()
    top_5_scores = top_5_results["average_similarity"].tolist()

    # Print top 5 similar files with similarity scores
    print("Top 5 similar files:")
    for name, score in zip(top_5_filenames, top_5_scores):
        print(f"{name} - Similarity Score: {score}")

    # Return list of tuples containing (filename, similarity_score)
    return list(zip(top_5_filenames, top_5_scores))


@app.route("/match", methods=["POST"])
def match():
    # Retrieve form data
    marketer_name = request.form.get("marketer_name")
    description = request.form.get("description")
    url = request.form.get("url")
    open_to = request.form.get("open_to")
    field = request.form.get("field").split(", ")  # Split field keywords by comma
    dr = request.form.get("dr")

    all_links_info = get_all_links(url)

    # Convert the list of dictionaries into a DataFrame
    df_links = pd.DataFrame(all_links_info)

    # Optional: Save the DataFrame to a CSV file
    if not os.path.exists("marketers"):
        os.makedirs("marketers")
    csv_file_path = os.path.join("marketers", f"{marketer_name}.csv")
    df_links.to_csv(csv_file_path, index=False)

    # Initialize results collection
    results = []

    try:
        # Perform URL request and create BeautifulSoup object
        url_request = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        urlSoup = BeautifulSoup(url_request.text, "lxml")

    except Exception as e:
        results.append(f"An error occurred: {e}")

    # Render the results in the response
    return render_template(
        "result.html",
        marketer_name=marketer_name,
        description=description,
        url=url,
        open_to=open_to,
        field=field,
        dr=dr,
        broken_links=seo_find_404(urlSoup),
        matched_marketers=match_new_marketers(description, field, df_links["title"]),
    )


agent = "Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:10.0) Gecko/20100101 Firefox/10.0"

if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists("output_files"):
        os.makedirs("output_files")

    app.run(debug=True)
