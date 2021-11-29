"""Main class of Topic2Facet."""
import logging

import numpy as np
import umap
import hdbscan
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine as cosine_distance
from sentence_transformers import SentenceTransformer


logger = logging.getLogger('Topic2Facet')
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class Topic2Facet:
    """Extract topics and facets from a corpus using sentence transformers."""
    def __init__(self,
                 embedding_model,
                 documents,
                 document_ids=None,
                 token_pattern=r"(?u)\b\w\w+\b",  # Default sklearn pattern
                 verbose=False
                 ):
        # General parameters
        # TODO: logging
        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
            logger.setLevel(logging.INFO)
            self.verbose = False

        # Store documents
        self.documents = np.array(documents, dtype="object")
        if document_ids is not None:
            self.document_ids = np.array(document_ids)
        else:
            self.document_ids = np.array(range(0, len(documents)))
        self.doc_idxs_to_ids = dict(enumerate(self.document_ids))

        # Preprocess vocabulary
        self.token_pattern = token_pattern
        self.vocab = self._preprocess_voc(self.documents, self.token_pattern)

        # Initialize embedding variables
        self.embedding_model = embedding_model
        self.document_vectors = None

        # Initialize topic/facet extraction variables
        self.topic_facets = {}
        self.doc_topic_facet = {}
        self.topic_extraction_parameters = {}
        self.facet_extraction_parameters = {}

        # Initialize history of topic merges
        self.merge_history = {}

    def save(self, file):
        """Save model to file."""
        joblib.dump(self, file)

    @staticmethod
    def load(file):
        """Load model from file."""
        model = joblib.load(file)
        return model

    def get_tf_attr(self, attr, topic=None, facet=None):
        """Query topic/facet attributes."""
        if topic == "all" and facet is None:
            topics_sorted = sorted(list(self.topic_facets.keys()))
            res = np.array([self.topic_facets[t][attr] for t in topics_sorted])
        elif topic in self.topic_facets and facet is None:
            res = self.topic_facets[topic][attr]
        elif topic in self.topic_facets and facet == "all":
            facets_sorted = sorted(list(self.topic_facets[topic]["facets"].keys()))
            res = np.array([self.topic_facets[topic]["facets"][f][attr]
                            for f in facets_sorted])
        elif topic in self.topic_facets and facet in self.topic_facets[topic]["facets"]:
            res = self.topic_facets[topic]["facets"][facet][attr]
        else:
            raise ValueError("Invalid combination of parameters provided.")
        return res

    def get_doc_topic(self):
        """Get document to topic mapping as an array."""
        return np.array([self.doc_topic_facet[idx]["topic"]
                         for idx in self.doc_topic_facet])

    def embed_corpus(self):
        """Compute document embeddings using a sentence-transfomer model."""
        model = SentenceTransformer(self.embedding_model)
        document_vectors = model.encode(self.documents,
                                        show_progress_bar=self.verbose)
        self.document_vectors = document_vectors
        return document_vectors

    def topic_extraction(self, n_components, n_neighbors, min_size,
                         min_samples=None, cluster_selection_method="eom",
                         n_words=30, random_state=None):
        """Extract and characterize high-level topics from the corpus."""
        if self.document_vectors is None:
            raise ValueError("Corpus embeddings must be computed prior to topic extraction.")

        # Dimension reduction
        umap_model = self._compute_umap(self.document_vectors,
                                        n_components,
                                        n_neighbors,
                                        random_state)
        document_vectors_ld = umap_model.embedding_

        # Topic clustering
        hdbscan_model = self._compute_hdbscan(document_vectors_ld,
                                              min_size,
                                              min_samples,
                                              cluster_selection_method=cluster_selection_method)
        clusters = hdbscan_model.labels_
        doc_topic_noisy = (clusters == -1)

        # Compute topic vectors
        topic_vectors = self._compute_topic_vectors(self.document_vectors,
                                                    hdbscan_model)

        # Assign noisy documents to topics
        doc_topic, doc_topic_sim = self._assign_noisy_docs(self.document_vectors,
                                                           topic_vectors,
                                                           clusters)

        # Reorder topics by decreasing size
        doc_topic, topic_vectors = self._reorder_topics(doc_topic, topic_vectors)
        __, topic_sizes = np.unique(doc_topic, return_counts=True)

        # Characterize topics by top N words with highest average TF-IDF
        # among the topic's documents subset
        topic_words, tfidf_vectors = self._topic_characterization(doc_topic,
                                                                  subset_docs=None,
                                                                  n_words=n_words)

        # Store general topic information
        self.topic_facets = {}
        for topic, size in enumerate(topic_sizes):
            doc_idxs = np.where(doc_topic == topic)[0]
            self.topic_facets[topic] = {
                "doc_idxs": doc_idxs,
                "size": size,
                "emb_vector": topic_vectors[topic],
                "tfidf_vector": tfidf_vectors[topic],
                "words": topic_words[topic],
                "facets": {}
            }

        # Store topic information for each document
        self.doc_topic_facet = {}
        for idx, __ in enumerate(doc_topic):
            self.doc_topic_facet[idx] = {"topic": doc_topic[idx],
                                         "topic_similarity": doc_topic_sim[idx],
                                         "topic_noisy": doc_topic_noisy[idx]
                                         }

        # Store parameters used for topic extraction
        self.topic_extraction_parameters = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_topic_size": min_size,
            "min_samples": min_samples,
            "n_words": n_words,
            "random_state": random_state
        }

        # Clean previously stored facet extraction parameters
        self.facet_extraction_parameters = None

        return self.doc_topic_facet, self.topic_facets

    def facet_extraction(self, n_components, n_neighbors, min_size,
                         min_samples=None, cluster_selection_method="eom",
                         n_words=30, random_state=None):
        """Extract and characterize facets (sub-topics) of each topic."""
        if self.doc_topic_facet == {}:
            raise ValueError("Topic extraction must be performed prior to facet extraction.")

        for topic in self.topic_facets:

            # Get topic's document vectors
            doc_idxs = [idx for idx in self.doc_topic_facet
                        if self.doc_topic_facet[idx]["topic"] == topic]
            document_vectors = self.document_vectors[doc_idxs]
            idxs_to_sub_idxs = {idx: i for i, idx in enumerate(doc_idxs)}
            sub_idxs_to_idxs = {i: idx for idx, i in idxs_to_sub_idxs.items()}

            # Dimension reduction
            umap_model = self._compute_umap(document_vectors,
                                            n_components,
                                            n_neighbors,
                                            random_state)
            document_vectors_ld = umap_model.embedding_

            # Facet clustering
            hdbscan_model = self._compute_hdbscan(document_vectors_ld,
                                                  min_size,
                                                  min_samples,
                                                  cluster_selection_method=cluster_selection_method)
            clusters = hdbscan_model.labels_
            doc_facet_noisy = (clusters == -1)

            # Compute facet vectors
            facet_vectors = self._compute_topic_vectors(document_vectors,
                                                        hdbscan_model)

            # Assign noisy documents to facets
            doc_facet, doc_facet_sim = self._assign_noisy_docs(document_vectors,
                                                               facet_vectors,
                                                               clusters)

            # Reorder topics by decreasing size
            doc_facet, facet_vectors = self._reorder_topics(doc_facet, facet_vectors)
            __, facet_sizes = np.unique(doc_facet, return_counts=True)

            # Characterize facets by top N words with highest average TF-IDF
            # among the facet's documents subset
            facet_words, tfidf_vectors = self._topic_characterization(doc_topic=doc_facet,
                                                                      subset_docs=doc_idxs,
                                                                      n_words=n_words)

            # Store facets information for each topic
            self.topic_facets[topic]["facets"] = {}
            for facet, size in enumerate(facet_sizes):
                facet_doc_idxs = np.array([sub_idxs_to_idxs[i]
                                           for i in np.where(doc_facet == facet)[0]])
                self.topic_facets[topic]["facets"][facet] = {"doc_idxs": facet_doc_idxs,
                                                             "size": size,
                                                             "emb_vector": facet_vectors[facet],
                                                             "tfidf_vector": tfidf_vectors[facet],
                                                             "words": facet_words[facet]
                                                             }

            # Store facet information for each document
            for idx in doc_idxs:
                idx_sub = idxs_to_sub_idxs[idx]
                self.doc_topic_facet[idx]["facet"] = doc_facet[idx_sub]
                self.doc_topic_facet[idx]["facet_similarity"] = doc_facet_sim[idx_sub]
                self.doc_topic_facet[idx]["facet_noisy"] = doc_facet_noisy[idx_sub]

        # Store parameters used for topic extraction
        self.facet_extraction_parameters = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_facet_size": min_size,
            "min_samples": min_samples,
            "n_words": n_words,
            "random_state": random_state
        }

        return self.doc_topic_facet, self.topic_facets

    def merge_topics(self, topics):
        """Merge two topics together."""
        # Order topics to be merged
        topics_to_merge = sorted(topics)
        topic1, topic2 = topics_to_merge

        size_topic1 = self.topic_facets[topic1]["size"]
        size_topic2 = self.topic_facets[topic2]["size"]
        size_merge = size_topic1 + size_topic2
        doc_idxs_merge = np.array((self.topic_facets[topic1]["doc_idxs"].tolist()
                                   + self.topic_facets[topic2]["doc_idxs"].tolist()))

        # Compute new mapping of topics
        new_topics_mapping = {}
        for topic, size in enumerate(self.get_tf_attr("size", "all")):
            if size > size_merge:
                new_topics_mapping[topic] = topic
                num_topic_merge = topic + 1
            else:
                if topic in [topic1, topic2]:
                    new_topics_mapping[topic] = num_topic_merge
                else:
                    if topic < topic1:
                        new_topics_mapping[topic] = topic + 1
                    elif topic < topic2:
                        new_topics_mapping[topic] = topic
                    else:
                        new_topics_mapping[topic] = topic - 1

        # Compute new doc-topic mapping
        doc_topic_new = [new_topics_mapping[topic] for topic in self.get_doc_topic()]

        # Compute new topic characterization
        new_t_words, new_t_tfidf = self._topic_characterization(doc_topic_new)
        words_merge = new_t_words[num_topic_merge]
        tfidf_merge = new_t_tfidf[num_topic_merge]

        # Compute new embedding vector
        idxs_vector_merge_compute = [idx for idx in self.doc_topic_facet
                                     if self.doc_topic_facet[idx]["topic"] in topics_to_merge
                                     and not self.doc_topic_facet[idx]["topic_noisy"]]
        emb_vector_merge = np.average(self.document_vectors[idxs_vector_merge_compute],
                                      axis=0)

        # Apply changes to model
        topic_facets_new = {}
        topic_facets_new[num_topic_merge] = {
                    "doc_idxs": doc_idxs_merge,
                    "size": size_merge,
                    "emb_vector": emb_vector_merge,
                    "tfidf_vector": tfidf_merge,
                    "words": words_merge
                }
        for old_topic in new_topics_mapping:
            if old_topic not in topics_to_merge:
                new_topic = new_topics_mapping[old_topic]
                topic_facets_new[new_topic] = self.topic_facets[old_topic]
                topic_facets_new[new_topic]["facets"] = {}   # Reinitialize facet detection
        self.topic_facets = topic_facets_new

        doc_topic_facet_new = {}
        for doc_idx in self.doc_topic_facet:
            if self.doc_topic_facet[doc_idx]["topic"] in topics_to_merge:
                doc_topic_facet_new[doc_idx] = {"topic": num_topic_merge}
                new_t_sim = 1 - cosine_distance(self.document_vectors[doc_idx],
                                                emb_vector_merge)
                doc_topic_facet_new[doc_idx]["topic_similarity"] = new_t_sim
                new_is_noisy = (self.doc_topic_facet[doc_idx]["topic_noisy"]
                                + self.doc_topic_facet[doc_idx]["topic_noisy"])
                doc_topic_facet_new[doc_idx]["topic_noisy"] = new_is_noisy
            else:
                old_topic = self.doc_topic_facet[doc_idx]["topic"]
                new_topic = new_topics_mapping[old_topic]
                topic_sim = self.doc_topic_facet[doc_idx]["topic_similarity"]
                topic_noisy = self.doc_topic_facet[doc_idx]["topic_noisy"]
                doc_topic_facet_new[doc_idx] = {"topic": new_topic,
                                                "topic_similarity": topic_sim,
                                                "topic_noisy": topic_noisy
                                                }
        self.doc_topic_facet = doc_topic_facet_new

        # Save operation into merge history
        if not self.merge_history:
            entry = 0
        else:
            entry = max(self.merge_history) + 1
        self.merge_history[entry] = {
            "topic1": topic1,
            "topic2": topic2,
            "old_new_mapping": new_topics_mapping
        }

        # Logging
        log = f"Topics {topic1} and {topic2} have been merged into topic {num_topic_merge}."
        logger.info(log)

    def get_post_merges_topic_mapping(self):
        """Get mapping from initial to final topics after merge(s)."""
        mappings = [self.merge_history[entry]["old_new_mapping"]
                    for entry in self.merge_history]

        full_mapping = {}
        for topic in mappings[0]:
            topic_new = topic   # Initialization
            for mapping in mappings:
                topic_new = mapping[topic_new]
            full_mapping[topic] = topic_new

        return full_mapping

    @staticmethod
    def _preprocess_voc(documents, token_pattern):
        """Extract individuals tokens from the corpus."""
        vectorizer = TfidfVectorizer(strip_accents="ascii",
                                     lowercase=True,
                                     token_pattern=token_pattern)
        vectorizer.fit(documents)
        vocab = vectorizer.get_feature_names()
        return np.array(vocab)

    @staticmethod
    def _compute_umap(document_vectors, n_components, n_neighbors,
                      random_state=None):
        """Compute low dimensional embeddings using the UMAP algorithm."""
        umap_model = umap.UMAP(n_neighbors=n_neighbors,
                               n_components=n_components,
                               min_dist=0,  # Maximize points density
                               metric='cosine',
                               low_memory=True,
                               random_state=random_state
                               )
        umap_model.fit(document_vectors)
        return umap_model

    @staticmethod
    def _compute_hdbscan(document_vectors, min_cluster_size, min_samples,
                         cluster_selection_method, prediction_data=False):
        """Perform density-based clustering using the HDBSCAN algorithm."""
        if min_samples is None:
            min_samples = min_cluster_size

        # Compute HDBSCAN clusters
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=min_samples,
                                        metric='euclidean',
                                        cluster_selection_method=cluster_selection_method,
                                        prediction_data=prediction_data
                                        )
        hdbscan_model.fit(document_vectors)

        return hdbscan_model

    @staticmethod
    def _compute_topic_vectors(document_vectors, hdbscan_model):
        """Compute a representative vector for each cluster found by HDBSCAN."""
        # Extract hdbscan elements
        clusters = hdbscan_model.labels_
        probs = hdbscan_model.probabilities_

        # Exclude noisy documents from topic vector computation
        # If all noisy (no clusters), include everything
        unique_labels = np.unique(clusters)
        if len(unique_labels) > 1 and -1 in unique_labels:
            unique_labels = unique_labels[1:]

        # Compute topic vectors as average of topic's document vectors,
        # weighted by hdbscan confidence score
        topic_vectors = []
        for topic in unique_labels:
            doc_idxs = np.where(clusters == topic)
            if all(probs == 0):
                # All docs noisy case
                topic_vec = np.average(document_vectors[doc_idxs],
                                       axis=0)
            else:
                # Normal case
                topic_vec = np.average(document_vectors[doc_idxs],
                                       weights=probs[doc_idxs],
                                       axis=0)
            topic_vectors.append(topic_vec)

        topic_vectors = np.array(topic_vectors)

        return topic_vectors

    @staticmethod
    def _assign_noisy_docs(document_vectors, topic_vectors, doc_topic):
        """Assign documents classified as noise by HDBSCAN to closest topic."""
        # Compute most similar topic for each document
        doc_topic_sim_mat = cosine_similarity(document_vectors, topic_vectors)
        most_sim_top = np.argmax(doc_topic_sim_mat, axis=1)

        # Assign noisy documents to closest topic
        doc_topic_new = []
        doc_topic_sim = []
        for i, clust in enumerate(doc_topic):
            if clust != -1:
                doc_topic_new.append(clust)
                doc_topic_sim.append(doc_topic_sim_mat[i, clust])
            else:
                new_topic = most_sim_top[i]
                doc_topic_new.append(new_topic)
                doc_topic_sim.append(doc_topic_sim_mat[i, new_topic])

        doc_topic_new = np.array(doc_topic_new)
        doc_topic_sim = np.array(doc_topic_sim)

        return doc_topic_new, doc_topic_sim

    @staticmethod
    def _reorder_topics(doc_topic, topic_vectors):
        """Reorder topics by decreasing size."""
        topics, sizes = np.unique(doc_topic, return_counts=True)
        topics_sorted = topics[np.flip(np.argsort(sizes))]
        mapping_dict = dict(zip(topics_sorted, topics))
        doc_topic_new = [mapping_dict[i] for i in doc_topic]
        topic_vectors_new = topic_vectors[topics_sorted]

        doc_topic_new = np.array(doc_topic_new)
        topic_vectors_new = np.array(topic_vectors_new)

        return doc_topic_new, topic_vectors_new

    def _topic_characterization(self, doc_topic, subset_docs=None, n_words=30):
        """Characterize each topic by the top n words wight highest tf-idf score."""
        if subset_docs is None:
            # In topic extraction, the corpus is the full collection of socuments
            corpus = self.documents
        else:
            # In facet extraction, the corpus is the topic's documents
            # A subset of document ids is passed to restrict the corpus
            corpus = self.documents[subset_docs]

        # Compute tf-idf matrix on the relevant corpus
        vectorizer = TfidfVectorizer(strip_accents="ascii",
                                     lowercase=True,
                                     token_pattern=self.token_pattern,
                                     stop_words="english"
                                     )
        tfidf_model = vectorizer.fit(corpus)
        words = np.array(tfidf_model.get_feature_names())

        topic_rep_words = []
        topic_vectors_tfidf = []
        for topic in np.unique(doc_topic):
            # Join all documents of the topic as a single string
            doc_idxs = np.where(doc_topic == topic)[0]
            big_doc_topic = " ".join(corpus[doc_idxs])
            # Compute tf-idf embedding of this meta -document
            topic_tfidf = vectorizer.transform([big_doc_topic]).toarray()[0]
            topic_vectors_tfidf.append(topic_tfidf)
            # Characterize topic with top n words wight highest tf-idf score
            top_scores = np.flip(np.argsort(topic_tfidf))
            top_words = words[top_scores][:n_words]
            topic_rep_words.append(top_words)

        topic_rep_words = np.array(topic_rep_words)
        topic_vectors_tfidf = np.array(topic_vectors_tfidf)

        return topic_rep_words, topic_vectors_tfidf
