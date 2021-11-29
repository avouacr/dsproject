import os
from datetime import datetime
import requests

import pandas as pd


def get_batch(sub, after, before, token):
    """Fetch a batch of submissions from the API."""
    url = (f"https://api.pushshift.io/reddit/search/submission/?subreddit={sub}"
           f"&access_token={token}"
           f"&after={after}&before={before}"
           "&size=500")
    req = requests.get(url)
    data = json.loads(req.text)
    return data['data']

def convert_datetime(dto):
    """Convert Python datetime object to timestamp and int YYYYMMDD."""
    date_timestamp = int(datetime.timestamp(dto))
    date_int = dto.year * 10000 + dto.month * 100 + dto.day
    return date_timestamp, date_int

def preprocess_self_text(text):
    """Remove recurrent noisy elements from self-texts."""
    text = text.replace("|", " ")  # Avoid pb with sep for csv
    text = re.sub(r"(_____\s+&gt.+)", " ", text)  # Remove footnote
    text = text.replace("[deleted]", " ").replace("[removed]", " ")
    text = text.strip()  # Remove excessing space at the end
    return text

def extract_info(subm):
    """Extract relevant info from the data dictionary of a submission."""
    title = subm['title']
    regexp_match = re.search(r"(CMV|cmv):\s?(.+)", title)
    if regexp_match:
        # Extract proper title
        title_sub = regexp_match.group(2).strip()
        if title_sub[-1] != ".":
            title_sub = title_sub + "."
        # Extract other relevant info
        subm_id = subm["id"]
        author = subm["author"]
        try:
            self_text = preprocess_self_text(subm["selftext"])
        except KeyError:
            self_text = ""
        timestamp = subm["created_utc"]
        nb_comments = subm["num_comments"]
        score = subm["score"]

        return subm_id, title_sub, author, timestamp, nb_comments, score, self_text

    return None

def get_all_titles(sub, start_timestamp, end_timestamp, token, path_save):
    """Extract all titles from a sub between two dates."""
    data = get_batch(sub, start_timestamp, end_timestamp, token)
    batch_data = []
    while data:
        # Get current batch timerange
        batch_start = str(datetime.fromtimestamp(data[0]['created_utc']))
        batch_end = str(datetime.fromtimestamp(data[-1]['created_utc']))

        # Extract data from current batch
        for submission in data:
            subm_data = extract_info(submission)
            if subm_data is not None:
                batch_data.append(subm_data)
        new_start_timestamp = data[-1]['created_utc']  # Start date of next batch

        print(f"Batch processed : {batch_start} - {batch_end}. "
              f"Next batch start timestamp : {new_start_timestamp}.")

        # Get next batch
        try:
            data = get_batch(sub, new_start_timestamp, end_timestamp, token)
        except json.JSONDecodeError:
            # Retry after sleep time
            time.sleep(5)
            try:
                data = get_batch(sub, new_start_timestamp, end_timestamp, token)
            except json.JSONDecodeError:
                print(f"Batch failed after retry : {batch_start} - {batch_end}.")
                continue
                
    df_titles = pd.DataFrame(data=batch_data)
                
    return df_titles
