Resume Analysis & Candidate Ranking System
This project is a Flask-based web application designed for analyzing resumes and ranking candidates based on their suitability for a given job description. It utilizes various natural language processing (NLP) techniques such as TF-IDF vectorization, cosine similarity, and Latent Dirichlet Allocation (LDA) for topic modeling.

Features
Resume Upload: Users can upload a ZIP file containing multiple resumes along with a PDF file containing the job description.

Resume Analysis: The application analyzes the content of each resume using TF-IDF vectorization to extract important keywords and information.

Job Role Analysis: It analyzes the job description using TF-IDF vectorization to identify key requirements and skills.

Topic Modeling: The application applies Latent Dirichlet Allocation (LDA) to both resumes and the job description to identify underlying topics.

Candidate Ranking: Resumes are ranked based on their similarity to the job description, taking into account both TF-IDF scores and LDA topic modeling.

Technologies Used
Flask: Python-based web framework for handling HTTP requests and responses.
NLTK (Natural Language Toolkit): Library for NLP tasks such as tokenization and stop word removal.
Gensim: Library for topic modeling and document similarity.
scikit-learn: Library for machine learning tasks including TF-IDF vectorization and cosine similarity.
PDFMiner & PyMuPDF: Libraries for extracting text from PDF documents.
HTML/CSS/JavaScript: Frontend development for user interaction.
Usage
Upload Resumes: Users upload a ZIP file containing resumes and a PDF file containing the job description.

Analysis: The application analyzes the resumes and the job description using NLP techniques.

Ranking: Resumes are ranked based on their similarity to the job description, and the leaderboard is displayed.

Setup
Install the required Python libraries using pip install -r requirements.txt.
Run the Flask application using python app.py.
Access the application through a web browser at http://localhost:5000.
Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/improvement).
Make your changes and commit them (git commit -am 'Add new feature').
Push the changes to your branch (git push origin feature/improvement).
Create a new Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
