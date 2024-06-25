from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
import pandas as pd
import fitz
from nltk.tokenize import word_tokenize
import zipfile
import os


app = Flask(__name__)


def load_data(job_description_file, resumes_file):
    resumes_df = []
    stop_words = set(stopwords.words('english'))
    for resume in resumes_file:
        text = ""
        with fitz.open(resume) as pdf_document:
            for page_number in range(pdf_document.page_count):
                page = pdf_document[page_number]
                text += page.get_text()
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
        print(text)
        resumes_df.append({'resume_name':resume, 'resume_text': text})
    resumes_df = pd.DataFrame(resumes_df)

    job_descriptions = ""
    with fitz.open(job_description_file) as job_description_file:
        for page_number in range(job_description_file.page_count):
            page = job_description_file[page_number]
            job_descriptions += page.get_text()
    words = word_tokenize(job_descriptions)
    words = [word.lower() for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    job_descriptions = ' '.join(words)

    return job_descriptions, resumes_df

# Job Role Analysis
def analyze_job_role(job_description, stop_words):
    stop_words.update(["role", "responsibilities", "skills", "experience", "qualifications"])  # Add additional custom stop words
    job_description = ' '.join([word for word in job_description.lower().split() if word not in stop_words])

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform([job_description])

    feature_names = tfidf_vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.toarray()[0]

    keyword_scores = dict(zip(feature_names, tfidf_scores))

    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    top_keywords = dict([(keyword,score) for keyword, score in sorted_keywords[:50]])

    return top_keywords

# Candidate Resume Analysis
def analyze_candidate_resume(resume, stop_words):
    stop_words.update(["phone", "email", "address", "linkedin", "github"])  # Add additional custom stop words
    resume = ' '.join([word for word in resume.lower().split() if word not in stop_words])

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform([resume])

    feature_names = tfidf_vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.toarray()[0]

    resume_features = dict(zip(feature_names, tfidf_scores))

    sorted_keywords = sorted(resume_features.items(), key=lambda x: x[1], reverse=True)
    
    resume_features = dict([(keyword,score) for keyword, score in sorted_keywords[:50]])

    return resume_features

# Topic Modeling with LDA
def apply_lda(texts):
    tokenized_texts = [text.lower().split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    num_topics = 5  # Adjust the number of topics based on your requirements
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
    corpus_lda = lda_model[corpus]
    
    return lda_model, corpus_lda, dictionary

# Matching and Scoring
def calculate_similarity(job_keywords, resume_features, lda_model, corpus_lda, dictionary):
    # Compute cosine similarity between TF-IDF vectors of job keywords and resume features
    tfidf_cosine_similarity = cosine_similarity([list(job_keywords.values())], [list(resume_features.values())])[0][0]

    # Convert resume text to LDA vector
    resume_lda_vector = lda_model[dictionary.doc2bow(resume_features.keys())]

    # Compute similarity between job LDA topics and resume LDA vector
    lda_similarity = 0
    for topic_id, topic_score in resume_lda_vector:
        lda_similarity += topic_score * corpus_lda[0][topic_id][1]  # Weighted similarity based on LDA scores

    # Combine both similarities (you can adjust the weights based on importance)
    final_score = 0.7 * tfidf_cosine_similarity + 0.3 * lda_similarity
    
    
    return int(final_score*100)

# Ranking System
def rank_candidates(job_description, resumes, stop_words):
    job_keywords = analyze_job_role(job_description, stop_words)
    lda_model, corpus_lda, dictionary = apply_lda(job_description)

    ranking_scores = []
    for resume in resumes["resume_text"][:]:
        resume_features = analyze_candidate_resume(resume, stop_words)
        score = calculate_similarity(job_keywords, resume_features, lda_model, corpus_lda, dictionary)
        ranking_scores.append(score)
    ranked_candidates = sorted(enumerate(ranking_scores), key=lambda x: x[1], reverse=True)
    return ranked_candidates

def clear_directory(directory):
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Check if it is a file
            if os.path.isfile(file_path) and filename!='.gitkeep':
                # Delete the file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the file upload and displaying results
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        resumes_file = []
        uploaded_file = request.files['resumeZip']
        jd_uploaded_file = request.files['jd']
        print(uploaded_file)
        zip_path = 'uploads/Code.zip'
        uploaded_file.save(zip_path)
        job_description_file = 'uploads/JD.pdf'
        jd_uploaded_file.save(job_description_file)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('uploads/Extracted/')
        resumes_file = ['uploads/Extracted/'+x for x in os.listdir('uploads/Extracted/') if x!='.gitkeep']
        print(resumes_file)

        job_description, resumes = load_data(job_description_file, resumes_file)
        stop_words = set(stopwords.words('english'))
        ranked_candidates = rank_candidates(job_description, resumes, stop_words)
        
        leaderboard_data = [(resumes.iloc[idx]['resume_name'].split("/")[-1], score) for idx, score in ranked_candidates]
        print(leaderboard_data)
        
        os.remove(job_description_file)
        clear_directory("uploads\\Extracted")

        return jsonify(leaderboard_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
