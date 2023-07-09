import pysrt
from moviepy.editor import VideoFileClip
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import nltk

# Step 1: Find cosine similarity of captions with query


def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token for token in tokens if token.lower() not in stop_words]

    # Join the tokens back into a single string
    preprocessed_text = " ".join(filtered_tokens)

    return preprocessed_text


def calculate_cosine_similarity(query, captions):
    # Preprocess query and captions
    preprocessed_query = preprocess_text(query)
    preprocessed_captions = [preprocess_text(caption) for caption in captions]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        [preprocessed_query] + preprocessed_captions)

    # Calculate cosine similarity between query and captions
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    return similarity_scores.flatten()

# Step 2: Find most similar part of captions


def cut_video(start_time, end_time, input_video_path, output_video_path, k):
    video = VideoFileClip(input_video_path).subclip(start_time, end_time)
    video.write_videofile(output_video_path+str(k)+".mp4", codec="libx264")


def find_top_3_moments(query, captions, timestamps, input_video_path, output_video_path):
    similarity_scores = calculate_cosine_similarity(query, captions)
    top_indices = np.argpartition(similarity_scores, -3)[-3:]
    top_scores = similarity_scores[top_indices]
    top_captions = [captions[i] for i in top_indices]
    top_timestamps = [timestamps[i] for i in top_indices]
    k = 0
    top_moments = []
    for caption, timestamp, score in zip(top_captions, top_timestamps, top_scores):
        start_time, end_time = timestamp
        cut_video(start_time, end_time, input_video_path, output_video_path, k)
        k = k + 1
        top_moments.append({
            'caption': caption,
            'start_time': start_time,
            'end_time': end_time,
            'output_video_path': output_video_path,
            'similarity_score': score
        })

    return top_moments


# Example usage
query = "Enemy"
# Replace with actual SRT file path
srt_file_path = "Captions/videoplayback.srt"
# Replace with actual video file path
input_video_path = "Dataset/song.mp4"
# Replace with desired output file path
output_video_path = "output/"

# Extract captions and timestamps from SRT file
captions = []
timestamps = []

subs = pysrt.open(srt_file_path)
for sub in subs:
    captions.append(sub.text)
    timestamps.append((
        sub.start.seconds + sub.start.minutes * 60 + sub.start.hours * 3600,
        sub.end.seconds + sub.end.minutes * 60 + sub.end.hours * 3600
    ))

top_3_moments = find_top_3_moments(query, captions, timestamps, input_video_path, output_video_path)
for moment in top_3_moments:
    print("Caption:", moment['caption'])
    print("Start Time:", moment['start_time'])
    print("End Time:", moment['end_time'])
    print("Output Video Path:", moment['output_video_path'])
    print("Similarity Score:", moment['similarity_score'])
    print()
