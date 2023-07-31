import pysrt
import heapq
from moviepy.editor import VideoFileClip
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import nltk
# nltk.download('all')



def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token for token in tokens if token.lower() not in stop_words]

    preprocessed_text = " ".join(filtered_tokens)
    return preprocessed_text


def calculate_cosine_similarity(query, captions):
    preprocessed_query = preprocess_text(query)
    preprocessed_captions = [preprocess_text(caption) for caption in captions]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        [preprocessed_query] + preprocessed_captions)

    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    return similarity_scores.flatten()


def find_most_similar_caption(query, captions):
    similarity_scores = calculate_cosine_similarity(query, captions)
    unique_indices = list(set(range(len(similarity_scores))))

    indices = heapq.nlargest(3, unique_indices, key=similarity_scores.__getitem__)
    final_score=[]
    for similar_index in indices:
        temp_score=[]
        similar_caption = captions[similar_index]
        similar_score = similarity_scores[similar_index]
        temp_score.append(similar_index)
        temp_score.append(similar_caption)
        temp_score.append(similar_score)
        final_score.append(temp_score)
    return final_score



def cut_video(start_time, end_time, input_video_path, output_video_path,k):
    video = VideoFileClip(input_video_path).subclip(start_time, end_time)
    video.write_videofile(output_video_path+str(k)+".mp4", codec="libx264")


query = "Enemy"
srt_file_path = "Captions/videoplayback.srt"
input_video_path = "Dataset/song.mp4"
output_video_path = "Output/"

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

    start_time1, end_time1 = timestamps[max([index[0]-1, 0])]
    start_time2, end_time2 = timestamps[min([
        index[0]+1, len(timestamps)-1])]
    cut_video(start_time1, end_time2, input_video_path, output_video_path,k)
    k=k+1
