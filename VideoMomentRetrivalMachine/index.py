import pysrt
import heapq
from moviepy.editor import VideoFileClip
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import nltk
# nltk.download('all')

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


def find_most_similar_caption(query, captions):
    similarity_scores = calculate_cosine_similarity(query, captions)
    unique_indices = list(set(range(len(similarity_scores))))

    # Use heapq to find the indices of the largest distinct elements
    indices = heapq.nlargest(3, unique_indices, key=similarity_scores.__getitem__)
    # max_similarity_index = np.argmax(similarity_scores)
    final_score=[]
    for similar_index in indices:
        temp_score=[]
        similar_caption = captions[similar_index]
        similar_score = similarity_scores[similar_index]
        temp_score.append(similar_index)
        temp_score.append(similar_caption)
        temp_score.append(similar_score)
        final_score.append(temp_score)
    # print(max_similarity_index)
    return final_score

# Step 3: Cut and provide the part of video for the most similar caption


def cut_video(start_time, end_time, input_video_path, output_video_path,k):
    video = VideoFileClip(input_video_path).subclip(start_time, end_time)
    video.write_videofile(output_video_path+str(k)+".mp4", codec="libx264")


# Example usage
query = "Enemy"
# Replace with actual SRT file path
srt_file_path = "Captions/videoplayback.srt"
# Replace with actual video file path
input_video_path = "Dataset/song.mp4"
# Replace with desired output file path
output_video_path = "Output/"
# Extract captions and timestamps from SRT file

captions = []
timestamps = []

subs = pysrt.open(srt_file_path)
for sub in subs:
    captions.append(sub.text)
    timestamps.append((sub.start.seconds + sub.start.minutes * 60 + sub.start.hours * 3600,
                       sub.end.seconds + sub.end.minutes * 60 + sub.end.hours * 3600))

final_score = find_most_similar_caption(
    query, captions)
k=0
for index in final_score:
    print("Most similar caption:", index[1])
    print("Similarity score:", index[2])

    # Assuming you have the start and end time for the most similar caption
    start_time1, end_time1 = timestamps[max([index[0]-1, 0])]
    start_time2, end_time2 = timestamps[min([
        index[0]+1, len(timestamps)-1])]
    cut_video(start_time1, end_time2, input_video_path, output_video_path,k)
    k=k+1