'''''
Brainstorming...

should have different preprocessing profiles in different functions
so that when imported into the simple_rag.ipynp, they can be compared and best one is chosen. 

thinking what kind of preprocessing needed, and whether it has to be suitable with NER.
Is NER needed? the identifiable entities are mainly characters. 

In one of the parahraphs in sparknotes_bluesteye.json, it is mentioned "the girls" without specifying their names,
as they're following context from previous paragraph also under the same summary. 
So for NER, it would have to be on whole summary/ analysis sections, and it can be placed as metadata for all the chunks in that section.

If placed on whole summary/ analysis sections and all NER added to metadata for all child chunks, 
then the case above would still not be solved. Many characters are mentioned and so all will be attached to all chunks so not very helpful...

options:
1- no NER needed; it is only used for Person anyway and the names will show up in text for retrieval.
2- Do overlap chunking, so that the context is preserved and the NER can be applied to each chunk which should identify the relevant entities only.
    --> if not for NER, do we still need overlap chunking? guess we'll have to test it in .ipynb

Shouldn't preprocessing be made after chunking?? The sequence I am finding that makes sense now:
chunking, NER over chunks, preprocessing AFTER to be able to lowercase, remove stop words, remove punct w/o being unable to sentence chunk...
'''
# 02_preprocessing.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Import document loader from step 1
from s01_documents import load_documents

# Download required NLTK resources if using stopwords/stemming/lemmatization
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK WordNet...")
    nltk.download('wordnet', quiet=True)


# Helper functions for text preprocessing
def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def whitespace_tokenize(text):
    return text.split()

def make_lowercase(text):
    return text.lower()

def remove_urls(text):
    return re.sub(r"http\S+|www\.\S+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    return re.sub(r'\d+', '', text)


# Main text preprocessing function
def preprocess_text(text,
                    lowercase=False,         # Set to False by default to preserve capitalization for RAG/LLM context
                    remove_url=True,
                    remove_punct=False,      # Set to False by default to keep sentence structure for embeddings
                    remove_num=False,
                    normalize_space=True,
                    remove_stop_words=False, # Kept False so LLM receives natural English
                    preserve_negation=True,
                    use_stemming=False,
                    use_lemmatization=False):

    if lowercase:
        text = make_lowercase(text)

    if remove_url:
        text = remove_urls(text)

    if remove_punct:
        text = remove_punctuation(text)

    if remove_num:
        text = remove_numbers(text)

    if remove_stop_words:
        tokens = whitespace_tokenize(text)
        custom_stopwords = set(stopwords.words('english'))

        if preserve_negation:
            # Remove common negation words from stopwords
            negation_words = {"not", "no", "nor", "never"}
            custom_stopwords = {word for word in custom_stopwords if word not in negation_words and not word.endswith("n't")}

        tokens = [word for word in tokens if word not in custom_stopwords]
        text = ' '.join(tokens)

    if use_stemming:
        stemmer = PorterStemmer()
        tokens = whitespace_tokenize(text)
        tokens = [stemmer.stem(word) for word in tokens]
        text = ' '.join(tokens)

    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = whitespace_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        text = ' '.join(tokens)

    if normalize_space:
        text = normalize_whitespace(text)

    return text


def preprocess_documents(
                        file_path="data/sparknotes_bluesteye.json", 
                        lowercase=False, 
                        remove_url=True, 
                        remove_punct=False, 
                        remove_num=False, 
                        normalize_space=True, 
                        remove_stop_words=False, 
                        preserve_negation=True, 
                        use_stemming=False, 
                        use_lemmatization=False):
    """
    Loads raw SparkNotes documents, applies text cleaning while preserving 
    chapter and type metadata, and returns the preprocessed document list.
    """
    raw_docs = load_documents(file_path=file_path)
    processed_docs = []
    
    for doc in raw_docs:
        cleaned_text = preprocess_text(text=doc["text"], 
                                       lowercase=False, 
                                       remove_url=True, 
                                       remove_punct=False, 
                                       remove_num=False, 
                                       normalize_space=True, 
                                       remove_stop_words=False, 
                                       preserve_negation=True, 
                                       use_stemming=False, 
                                       use_lemmatization=False)
        
        # Keep documents that retain meaningful text after cleaning
        if len(cleaned_text) > 20:
            doc_copy = doc.copy()
            doc_copy["text"] = cleaned_text
            processed_docs.append(doc_copy)
            
    return processed_docs


if __name__ == "__main__":
    docs = preprocess_documents()
    print(f"Successfully preprocessed {len(docs)} documents.")
    if docs:
        print("\nSample Preprocessed Document:")
        print(f"ID: {docs[0]['id']} | Chapter: {docs[0]['chapter']} | Type: {docs[0]['type']}")
        print(f"Text: {docs[0]['text'][:120]}...")