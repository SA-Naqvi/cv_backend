from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import time
import re
import os
from utils import bruteForce, rabinKarp, kmp_matcher

app = Flask(__name__)
# Allow CORS from GitHub Pages domain and localhost for development
CORS(app, origins=[
    "https://sa-naqvi.github.io",
    "https://SA-Naqvi.github.io",
    "http://localhost:3000",
    "http://localhost:3001"
])

# Load dataset at startup
try:
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    print(f"✓ Dataset loaded: {len(dataset)} documents")
except FileNotFoundError:
    print("⚠ Warning: dataset.pkl not found. Using empty dataset.")
    dataset = {}

# Load job descriptions
job_descriptions = {}
job_desc_path = 'job_descriptions'
if os.path.exists(job_desc_path):
    for filename in os.listdir(job_desc_path):
        if filename.endswith('.txt'):
            job_name = filename.replace('.txt', '').replace('_', ' ').title()
            with open(os.path.join(job_desc_path, filename), 'r', encoding='utf-8') as f:
                keywords = f.read().strip()
                job_descriptions[job_name] = keywords
    if job_descriptions:
        print(f"✓ Loaded {len(job_descriptions)} job descriptions: {list(job_descriptions.keys())}")

# Common stop words to filter out
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
}

def remove_stop_words(text):
    """Remove stop words and punctuation from search query"""
    # Remove punctuation and convert to lowercase
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split and filter stop words
    words = [w.strip() for w in text_clean.split() if w.strip() and w not in STOP_WORDS]
    return ' '.join(words)

def calculate_match_score(text, keywords, algorithm='bruteForce', track_comparisons=False, track_collisions=False):
    """Calculate match score using specified algorithm - searches each keyword separately
    Returns: (score, all_matches, matched_keywords, missing_keywords, total_comparisons, total_collisions, file_size_category)"""
    if not keywords or not text:
        return 0, [], [], [], 0, 0, 'unknown'
    
    text_lower = text.lower()
    
    # Categorize file size
    text_size = len(text)
    if text_size < 1000:
        size_category = 'small'
    elif text_size > 10000:
        size_category = 'large'
    else:
        size_category = 'medium'
    
    # Split keywords into individual words
    keyword_list = [kw.strip() for kw in keywords.split() if kw.strip()]
    if not keyword_list:
        return 0, [], [], keyword_list, 0, 0, size_category
    
    all_matches = []
    matched_keywords = []
    total_comparisons = 0
    total_collisions = 0
    total_matches_count = 0
    
    # Search for each keyword separately
    for keyword in keyword_list:
        keyword_lower = keyword.lower()
        
        # Choose algorithm
        if algorithm == 'bruteForce':
            matches, comparisons = bruteForce(text_lower, keyword_lower)
            collisions = 0
        elif algorithm == 'rabinKarp':
            matches, comparisons, collisions = rabinKarp(text_lower, keyword_lower)
        elif algorithm == 'kmp':
            matches, comparisons = kmp_matcher(text_lower, keyword_lower)
            collisions = 0
        else:
            matches, comparisons, collisions = [], 0, 0
        
        if track_comparisons:
            total_comparisons += comparisons
        if track_collisions:
            total_collisions += collisions
        
        if matches:  # If keyword found
            matched_keywords.append(keyword)
            total_matches_count += len(matches)
            all_matches.extend([(keyword, pos) for pos in matches])
    
    # Calculate missing keywords
    missing_keywords = [kw for kw in keyword_list if kw not in matched_keywords]
    
    # Calculate relevance score: (matched keywords / total keywords) * 100
    # If all keywords are found at least once, score is 100%
    score = (len(matched_keywords) / len(keyword_list)) * 100 if keyword_list else 0
    score = round(score, 2)
    
    return score, all_matches, matched_keywords, missing_keywords, total_comparisons, total_collisions, size_category

@app.route('/search', methods=['POST'])
def search():
    """Search documents using keywords"""
    try:
        data = request.get_json()
        keywords = data.get('keywords', '')
        algorithm = data.get('algorithm', 'bruteForce')
        
        if not keywords:
            return jsonify({'error': 'Keywords required'}), 400
        
        # Remove stop words
        cleaned_keywords = remove_stop_words(keywords)
        if not cleaned_keywords:
            return jsonify({'error': 'No valid keywords after filtering stop words'}), 400
        
        # Search all documents
        results = []
        for filename, content in dataset.items():
            score, all_matches, matched_keywords, missing_keywords, comparisons, collisions, size_category = calculate_match_score(
                content, cleaned_keywords, algorithm, track_comparisons=True, track_collisions=(algorithm == 'rabinKarp')
            )
            if score > 0:  # Only include documents with matches
                results.append({
                    'filename': filename,
                    'score': score,
                    'matches': len(all_matches),
                    'matched_keywords': matched_keywords,
                    'missing_keywords': missing_keywords,
                    'total_keywords': len(cleaned_keywords.split()),
                    'comparisons': comparisons,
                    'collisions': collisions if algorithm == 'rabinKarp' else 0,
                    'size_category': size_category
                })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'results': results,
            'query': keywords,
            'cleaned_query': cleaned_keywords,
            'total_documents': len(dataset),
            'matched_documents': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Compare execution times of all three algorithms"""
    try:
        data = request.get_json()
        keywords = data.get('keywords', '')
        
        if not keywords:
            return jsonify({'error': 'Keywords required'}), 400
        
        # Remove stop words
        cleaned_keywords = remove_stop_words(keywords)
        if not cleaned_keywords:
            return jsonify({'error': 'No valid keywords after filtering stop words'}), 400
        
        algorithms = {
            'Brute Force': 'bruteForce',
            'Rabin-Karp': 'rabinKarp',
            'KMP': 'kmp'
        }
        
        comparison_results = []
        
        # Test each algorithm
        for algo_name, algo_key in algorithms.items():
            start_time = time.perf_counter()
            total_comparisons = 0
            total_collisions = 0
            small_cv_count = 0
            medium_cv_count = 0
            large_cv_count = 0
            
            # Run algorithm on all documents
            results = []
            for filename, content in dataset.items():
                score, all_matches, matched_keywords, missing_keywords, comparisons, collisions, size_category = calculate_match_score(
                    content, cleaned_keywords, algo_key, track_comparisons=True, track_collisions=(algo_key == 'rabinKarp')
                )
                total_comparisons += comparisons
                if algo_key == 'rabinKarp':
                    total_collisions += collisions
                
                if score > 0:
                    if size_category == 'small':
                        small_cv_count += 1
                    elif size_category == 'medium':
                        medium_cv_count += 1
                    elif size_category == 'large':
                        large_cv_count += 1
                    
                    results.append({
                        'filename': filename,
                        'score': score,
                        'matches': len(all_matches),
                        'matched_keywords': matched_keywords,
                        'missing_keywords': missing_keywords,
                        'size_category': size_category
                    })
            
            end_time = time.perf_counter()
            execution_time = round((end_time - start_time) * 1000, 3)  # Convert to ms
            
            # Sort results
            results.sort(key=lambda x: x['score'], reverse=True)
            
            comparison_results.append({
                'algorithm': algo_name,
                'execution_time': execution_time,
                'total_comparisons': total_comparisons,
                'total_collisions': total_collisions if algo_key == 'rabinKarp' else 0,
                'matched_documents': len(results),
                'small_cv_count': small_cv_count,
                'medium_cv_count': medium_cv_count,
                'large_cv_count': large_cv_count,
                'top_results': results[:5]  # Return top 5 matches
            })
        
        return jsonify({
            'query': keywords,
            'cleaned_query': cleaned_keywords,
            'comparisons': comparison_results,
            'total_documents': len(dataset)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/job_descriptions', methods=['GET'])
def get_job_descriptions():
    """Get available job descriptions"""
    return jsonify({
        'job_descriptions': {name: keywords.split('\n') for name, keywords in job_descriptions.items()}
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'documents_loaded': len(dataset),
        'job_descriptions_loaded': len(job_descriptions)
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)