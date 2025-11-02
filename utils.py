import os
import fitz  # PyMuPDF
import pickle
import zipfile
from xml.etree.ElementTree import XML
from tqdm import tqdm

WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PARA = WORD_NAMESPACE + 'p'
TEXT = WORD_NAMESPACE + 't'

def get_docx_text(path):
    """Extract text from DOCX file"""
    try:
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        document.close()
        tree = XML(xml_content)
        paragraphs = []
        for paragraph in tree.iter(PARA):
            texts = [node.text for node in paragraph.iter(TEXT) if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        return '\n\n'.join(paragraphs)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def extract_pdf_text(path):
    """Extract text from PDF file"""
    text = []
    try:
        doc = fitz.open(path)
        for page in doc:
            text.append(page.get_text("text"))
        doc.close()
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return "\n".join(text)

def bruteForce(string, pattern):
    """Brute force string matching algorithm - O(n*m) worst case
    Returns: (matches, comparison_count)"""
    n = len(string)
    m = len(pattern)
    if m > n or m == 0:
        return [], 0
    
    string = string.lower()
    pattern = pattern.lower()
    idx = []
    comparisons = 0
    
    for i in range(n - m + 1):
        j = 0
        while j < m:
            comparisons += 1  # Count every comparison
            if string[i + j] != pattern[j]:
                break
            j += 1
        if j == m:
            idx.append(i)
    
    return idx, comparisons


def rabinKarp(string, pattern):
    """Optimized Rabin-Karp with faster rolling hash
    Returns: (matches, comparison_count, collision_count)"""
    n = len(string)
    m = len(pattern)
    if m > n or m == 0:
        return [], 0, 0
    
    base = 256
    mod = 7919
    string = string.lower()
    pattern = pattern.lower()
    
    # Precompute base^(m-1) mod mod
    h = 1
    for _ in range(m - 1):
        h = (h * base) % mod
    
    # Calculate initial hashes
    hashPattern = 0
    hashString = 0
    for i in range(m):
        hashPattern = (hashPattern * base + ord(pattern[i])) % mod
        hashString = (hashString * base + ord(string[i])) % mod
    
    idx = []
    comparisons = 0
    collisions = 0
    
    for i in range(n - m + 1):
        comparisons += 1
        if hashPattern == hashString:
            # Verify match - count each character comparison
            is_match = True
            for k in range(m):
                comparisons += 1
                if string[i + k] != pattern[k]:
                    is_match = False
                    break
            
            if is_match:
                idx.append(i)
            else:
                # Hash matched but strings don't - this is a collision
                collisions += 1
        
        # Rolling hash for next window
        if i < n - m:
            hashString = (hashString - ord(string[i]) * h) % mod
            hashString = (hashString * base + ord(string[i + m])) % mod
            hashString = (hashString + mod) % mod
    
    return idx, comparisons, collisions


def prefix(p):
    """Compute prefix function for KMP algorithm
    Returns: (prefix_array, comparison_count)"""
    m = len(p)
    Pi = [0] * m
    k = 0
    comparisons = 0
    
    for q in range(1, m):
        while k > 0 and p[k] != p[q]:
            comparisons += 1  # Count the failed comparison
            k = Pi[k - 1]
        
        comparisons += 1  # Count the comparison p[k] == p[q]
        if p[k] == p[q]:
            k += 1
        
        Pi[q] = k
    
    return Pi, comparisons


def kmp_matcher(S, P):
    """KMP string matching algorithm
    Returns: (matches, comparison_count)"""
    n = len(S)
    m = len(P)
    if m == 0 or m > n:
        return [], 0
    
    Pi, prefix_comparisons = prefix(P)
    q = 0
    matches = []
    comparisons = prefix_comparisons
    
    S = S.lower()
    P = P.lower()
    
    for i in range(n):
        while q > 0 and P[q] != S[i]:
            comparisons += 1  # Count the failed comparison
            q = Pi[q - 1]
        
        comparisons += 1  # Count the comparison P[q] == S[i]
        if P[q] == S[i]:
            q += 1
        
        if q == m:
            matches.append(i - m + 1)
            q = Pi[q - 1]
    
    return matches, comparisons


def build_dataset(folder_path, output_file='dataset.pkl'):
    """
    Build a dataset from documents in a folder
    Supports PDF and DOCX files
    """
    dataset = {}
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found")
        return
    
    files = [f for f in os.listdir(folder_path) 
             if f.endswith(('.pdf', '.docx'))]
    
    print(f"Processing {len(files)} files...")
    
    for filename in tqdm(files):
        filepath = os.path.join(folder_path, filename)
        
        if filename.endswith('.pdf'):
            text = extract_pdf_text(filepath)
        elif filename.endswith('.docx'):
            text = get_docx_text(filepath)
        else:
            continue
        
        if text.strip():
            dataset[filename] = text
    
    # Save dataset
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"✓ Dataset created: {len(dataset)} documents saved to {output_file}")
    return dataset

if __name__ == "__main__":
    # Example usage: python utils.py
    # This will create a dataset.pkl from documents in 'documents' folder
    import sys
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = 'documents'
    
    build_dataset(folder)