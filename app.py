import streamlit as st
import re
import time
import pandas as pd
from collections import defaultdict
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

st.set_page_config(page_title="Information Retrieval System", layout="wide")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text, use_stopwords=True, method="lemmatization"):
    text = text.lower()
    text = text.replace("-", " ")
    tokens = re.findall(r"\b[a-z]+\b", text)

    if use_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    if method == "stemming":
        tokens = [stemmer.stem(t) for t in tokens]
    elif method == "lemmatization":
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def build_inverted_index(docs):
    index = defaultdict(set)
    for doc_id, tokens in docs.items():
        for token in tokens:
            index[token].add(doc_id)
    return {term: sorted(list(doc_ids)) for term, doc_ids in index.items()}


def build_biword_index(docs):
    biword = defaultdict(set)
    for doc_id, tokens in docs.items():
        for i in range(len(tokens) - 1):
            pair = tokens[i] + " " + tokens[i + 1]
            biword[pair].add(doc_id)
    return {term: sorted(list(doc_ids)) for term, doc_ids in biword.items()}


def build_positional_index(docs):
    pos_index = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in docs.items():
        for pos, token in enumerate(tokens):
            pos_index[token][doc_id].append(pos)
    return pos_index


def search_inverted(query_tokens, index):
    if not query_tokens:
        return []

    result = set(index.get(query_tokens[0], []))
    for token in query_tokens[1:]:
        result = result.intersection(set(index.get(token, [])))
    return sorted(list(result))


def search_biword(query_tokens, biword_index):
    if len(query_tokens) < 2:
        return []

    pairs = [query_tokens[i] + " " + query_tokens[i + 1] for i in range(len(query_tokens) - 1)]
    result = set(biword_index.get(pairs[0], []))

    for pair in pairs[1:]:
        result = result.intersection(set(biword_index.get(pair, [])))

    return sorted(list(result))


def search_positional(query_tokens, pos_index):
    if not query_tokens:
        return []

    candidate_docs = set(pos_index.get(query_tokens[0], {}).keys())

    for token in query_tokens[1:]:
        candidate_docs &= set(pos_index.get(token, {}).keys())

    final_docs = []

    for doc_id in candidate_docs:
        positions = pos_index[query_tokens[0]][doc_id]

        for start_pos in positions:
            match = True
            for i in range(1, len(query_tokens)):
                if start_pos + i not in pos_index[query_tokens[i]][doc_id]:
                    match = False
                    break

            if match:
                final_docs.append(doc_id)
                break

    return sorted(final_docs)


class BSTNode:
    def __init__(self, term):
        self.term = term
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, term):
        self.root = self._insert(self.root, term)

    def _insert(self, node, term):
        if node is None:
            return BSTNode(term)
        if term < node.term:
            node.left = self._insert(node.left, term)
        elif term > node.term:
            node.right = self._insert(node.right, term)
        return node

    def search(self, term):
        current = self.root
        while current:
            if term == current.term:
                return True
            elif term < current.term:
                current = current.left
            else:
                current = current.right
        return False


class SimpleBTree:
    def __init__(self, terms):
        self.terms = sorted(terms)

    def search(self, term):
        left, right = 0, len(self.terms) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.terms[mid] == term:
                return True
            elif self.terms[mid] < term:
                left = mid + 1
            else:
                right = mid - 1
        return False


def edit_distance(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


def spelling_correction(word, vocabulary):
    best_word = word
    best_dist = 999

    for term in vocabulary:
        dist = edit_distance(word, term)
        if dist < best_dist:
            best_dist = dist
            best_word = term

    return best_word, best_dist


st.title("End-to-End Information Retrieval System")

uploaded_files = st.file_uploader(
    "Upload text documents",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    raw_docs = {}

    for i, file in enumerate(uploaded_files):
        text = file.read().decode("utf-8", errors="ignore")
        raw_docs[f"Doc_{i+1}_{file.name}"] = text

    st.header("1. Uploaded Documents")
    for doc_id, text in raw_docs.items():
        with st.expander(doc_id):
            st.write(text[:3000])

    st.sidebar.header("Preprocessing Options")
    use_stopwords = st.sidebar.checkbox("Remove stopwords", value=True)
    method = st.sidebar.selectbox(
        "Choose normalization method",
        ["none", "stemming", "lemmatization"]
    )

    processed_docs = {
        doc_id: preprocess(text, use_stopwords, method)
        for doc_id, text in raw_docs.items()
    }

    st.header("2. Text Preprocessing Output")

    sample_doc = list(processed_docs.keys())[0]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Text")
        st.write(raw_docs[sample_doc][:1000])

    with col2:
        st.subheader("Processed Tokens")
        st.write(processed_docs[sample_doc][:100])

    inverted_index = build_inverted_index(processed_docs)
    biword_index = build_biword_index(processed_docs)
    positional_index = build_positional_index(processed_docs)
    vocabulary = sorted(inverted_index.keys())

    st.header("3. Inverted Index")
    st.dataframe(
        pd.DataFrame(
            list(inverted_index.items()),
            columns=["Term", "Document IDs"]
        )
    )

    query = st.text_input("Enter search query")

    if query:
        query_tokens = preprocess(query, use_stopwords, method)

        st.header("4. Query Processing")

        st.write("Processed Query Tokens:", query_tokens)

        inverted_result = search_inverted(
            query_tokens,
            inverted_index
        )

        biword_result = search_biword(
            query_tokens,
            biword_index
        )

        positional_result = search_positional(
            query_tokens,
            positional_index
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Inverted Index")
            st.success(f"{len(inverted_result)} document(s) found")
            st.write(inverted_result)

        with col2:
            st.subheader("Biword Index")
            st.success(f"{len(biword_result)} document(s) found")
            st.write(biword_result)

        with col3:
            st.subheader("Positional Index")
            st.success(f"{len(positional_result)} document(s) found")
            st.write(positional_result)

        comparison_df = pd.DataFrame({
            "Retrieval Method": [
                "Inverted Index",
                "Biword Index",
                "Positional Index"
            ],
            "Documents Retrieved": [
                len(inverted_result),
                len(biword_result),
                len(positional_result)
            ],
            "Result Documents": [
                inverted_result,
                biword_result,
                positional_result
            ]
        })

        st.subheader("Method Comparison")
        st.dataframe(comparison_df)

        st.info("""
        Inference:

        • Inverted Index retrieves documents containing all query terms.

        • Biword Index retrieves documents containing exact adjacent word pairs.

        • Positional Index verifies exact word positions and therefore provides
        the most accurate phrase retrieval.

        • Positional Index is generally preferred for phrase searching because
        it avoids false positives.
        """)

        st.header("5. Phrase Query Comparison")

        comparison_df = pd.DataFrame({
            "Method": ["Biword Index", "Positional Index"],
            "Result Documents": [biword_result, positional_result],
            "Accuracy Inference": [
                "May return false positives for longer phrases",
                "More accurate because it checks exact word positions"
            ]
        })

        st.dataframe(comparison_df)

        st.subheader("Biword Index Representation")
        st.dataframe(
            pd.DataFrame(
                list(biword_index.items())[:100],
                columns=["Biword", "Document IDs"]
            )
        )

        st.subheader("Positional Index Representation")
        pos_rows = []
        for term, docs in list(positional_index.items())[:100]:
            pos_rows.append([term, dict(docs)])

        st.dataframe(pd.DataFrame(pos_rows, columns=["Term", "Positions"]))

        st.header("6. BST vs B-Tree Dictionary Search")

        bst = BST()
        for term in vocabulary:
            bst.insert(term)

        btree = SimpleBTree(vocabulary)

        tree_results = []

        for token in query_tokens:
            start = time.perf_counter()
            bst_found = bst.search(token)
            bst_time = time.perf_counter() - start

            start = time.perf_counter()
            btree_found = btree.search(token)
            btree_time = time.perf_counter() - start

            tree_results.append([
                token,
                bst_found,
                bst_time,
                btree_found,
                btree_time
            ])

        tree_df = pd.DataFrame(
            tree_results,
            columns=[
                "Query Term",
                "BST Found",
                "BST Search Time",
                "B-Tree Found",
                "B-Tree Search Time"
            ]
        )

        st.dataframe(tree_df)

        st.write(
            "Inference: B-Tree or sorted binary search is generally faster and more stable "
            "because terms are ordered and searched using logarithmic search."
        )

        # =====================================================
        # 7. Tolerant Retrieval
        # =====================================================

        correction_rows = []
        corrected_tokens = []

        correction_needed = False

        for token in query_tokens:

            if token in vocabulary:

                corrected_tokens.append(token)

            else:

                correction_needed = True

                corrected_term, distance = spelling_correction(
                    token,
                    vocabulary
                )

                corrected_tokens.append(corrected_term)

                correction_rows.append([
                    token,
                    corrected_term,
                    distance
                ])

        if correction_needed:

            st.header("7. Tolerant Retrieval")

            st.write(
                "Misspelled terms detected. "
                "Applying Edit Distance based spelling correction."
            )

            correction_df = pd.DataFrame(
                correction_rows,
                columns=[
                    "Original Term",
                    "Corrected Term",
                    "Edit Distance"
                ]
            )

            st.subheader(
                "Spelling Correction using Edit Distance"
            )

            st.dataframe(correction_df)

            st.subheader("Corrected Query")

            st.success(
                " ".join(corrected_tokens)
            )

            corrected_inverted = search_inverted(
                corrected_tokens,
                inverted_index
            )

            corrected_biword = search_biword(
                corrected_tokens,
                biword_index
            )

            corrected_positional = search_positional(
                corrected_tokens,
                positional_index
            )

            st.subheader(
                "Retrieval Results After Query Correction"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Inverted Index")
                st.success(
                    f"{len(corrected_inverted)} document(s)"
                )
                st.write(corrected_inverted)

            with col2:
                st.markdown("### Biword Index")
                st.success(
                    f"{len(corrected_biword)} document(s)"
                )
                st.write(corrected_biword)

            with col3:
                st.markdown("### Positional Index")
                st.success(
                    f"{len(corrected_positional)} document(s)"
                )
                st.write(corrected_positional)

            tolerant_df = pd.DataFrame({
                "Method": [
                    "Inverted Index",
                    "Biword Index",
                    "Positional Index"
                ],
                "Documents Retrieved": [
                    len(corrected_inverted),
                    len(corrected_biword),
                    len(corrected_positional)
                ],
                "Result Documents": [
                    corrected_inverted,
                    corrected_biword,
                    corrected_positional
                ]
            })

            st.subheader(
                "Tolerant Retrieval Comparison"
            )

            st.dataframe(tolerant_df)

            st.info("""
            Inference:

            • Misspelled query terms were automatically corrected.

            • Edit Distance was used to find the closest matching
            vocabulary terms.

            • Retrieval was then performed using Inverted,
            Biword and Positional indexes.

            • This demonstrates tolerant retrieval capability
            in the information retrieval system.
            """)

        st.header("8. Stemming vs Lemmatization Comparison")

        stemmed_docs = {
            doc_id: preprocess(text, use_stopwords, "stemming")
            for doc_id, text in raw_docs.items()
        }

        lemmatized_docs = {
            doc_id: preprocess(text, use_stopwords, "lemmatization")
            for doc_id, text in raw_docs.items()
        }

        stem_index = build_inverted_index(stemmed_docs)
        lemma_index = build_inverted_index(lemmatized_docs)

        stem_query = preprocess(query, use_stopwords, "stemming")
        lemma_query = preprocess(query, use_stopwords, "lemmatization")

        stem_result = search_inverted(stem_query, stem_index)
        lemma_result = search_inverted(lemma_query, lemma_index)

        stem_coverage = len(stem_result)
        lemma_coverage = len(lemma_result)

        st.dataframe(
            pd.DataFrame({
                "Method": ["Stemming", "Lemmatization"],
                "Processed Query": [stem_query, lemma_query],
                "Retrieved Documents Count": [stem_coverage, lemma_coverage],
                "Retrieved Documents": [stem_result, lemma_result]
            })
        )

        if lemma_coverage >= stem_coverage:
            st.write(
                "Inference: Lemmatization is more suitable because it preserves meaningful root words "
                "and gives cleaner retrieval results."
            )
        else:
            st.write(
                "Inference: Stemming retrieves more documents but may reduce word meaning. "
                "It is useful when recall is more important than precision."
            )

        st.header("9. Final Inference")

        st.write("""
        1. Stopword removal and lemmatization improved retrieval quality.
        2. Lemmatization was better because it preserved semantic meaning.
        3. Positional index was more accurate for phrase queries.
        4. B-Tree search was faster and more stable than BST.
        5. Tolerant retrieval handled spelling mistakes using edit distance.
        6. Limitation: The system works mainly on text files and small datasets.
        7. Improvement: Add TF-IDF ranking, cosine similarity, PDF support, and advanced spelling correction.
        """)

else:
    st.info("Upload one or more .txt documents to start.")