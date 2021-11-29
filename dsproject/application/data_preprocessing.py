""""""
import os
import sys
import joblib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

PROJECT_PATH = "/home/romain/projects/socsemics"
sys.path.append(PROJECT_PATH)

import experiments.scripts.topic_modelling.st_tm.st_tm as st_topic_model
import experiments.scripts.topic_modelling.st_tm.helpers as helpers

DATA_DIR = os.path.join(PROJECT_PATH, "experiments", "data")
MODELS_DIR = os.path.join(PROJECT_PATH, "experiments", "models")
TM_DIR = os.path.join(MODELS_DIR, "topic_modelling", "st_tm")
APP_DIR = os.path.join(PROJECT_PATH, "experiments", "viz_app")


def compute_topic_hulls(model, umap_model_2d, eps=0.1, min_samples=75,
                        plot=False):
    """"""
    n_topics = len(np.unique(model.doc_topic))
    palette = plt.get_cmap("Spectral")(np.linspace(0, 1, n_topics))

    topic_hulls = []
    for topic in np.unique(model.doc_topic):
        docs_idxs = np.where(model.doc_topic == topic)[0]
        t_points = umap_model_2d.embedding_[docs_idxs]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(t_points)

        t_hulls = []
        for label in np.unique(clustering.labels_):
            if label != -1:
                island = t_points[clustering.labels_ == label]
                hull = island[ConvexHull(island).vertices]
                t_hulls.append(hull)
        topic_hulls.append(t_hulls)

    if plot:
        axs = helpers.plot_topics(model, mark_noisy=False,
                                  umap_model_2d=umap_model_2d,
                                  show_legend=False)
        for topic, t_hulls in enumerate(topic_hulls):
            for hull in t_hulls:
                axs.plot(hull[:, 0], hull[:, 1],
                         color=to_hex(palette[topic]), linestyle='dashed',
                         linewidth=3, markersize=8)

    return topic_hulls


def hulls_to_coords(hulls):
    """"""
    x_coords = []
    y_coords = []
    for __, t_hulls in enumerate(hulls):
        t_xs = []
        t_ys = []
        for hull in t_hulls:
            t_xs.append([list(hull[:, 0])])
            t_ys.append([list(hull[:, 1])])
        x_coords.append(t_xs)
        y_coords.append(t_ys)

    return x_coords, y_coords


if __name__ == "__main__":

    # Import trained topic/facet model
    MODEL_NAME = "st_tm_cmv_titles_distilroberta-base-paraphrase-v1"
    PATH_MODEL = os.path.join(TM_DIR, MODEL_NAME)
    model = st_topic_model.STTopicModel.load(PATH_MODEL)

    # Import 2d embeddings
    N_NEIGHBORS = model.topic_extraction_parameters["n_neighbors"]
    UMAP_2D_NAME = f"umap_2d_{N_NEIGHBORS}_neighbors"
    PATH_UMAP_2D = os.path.join(TM_DIR, UMAP_2D_NAME)
    umap_model_2d = joblib.load(PATH_UMAP_2D)

    # Compute topic polygons
    topic_hulls = compute_topic_hulls(model, eps=0.1, min_samples=75, plot=False,
                                      umap_model_2d=umap_model_2d)
    topic_polygons_xs, topic_polygons_ys = hulls_to_coords(topic_hulls)

    # Store app input data
    app_data = {}
    app_data["umap_2d_embeddings"] = umap_model_2d.embedding_
    app_data["documents"] = model.documents
    app_data["document_ids"] = model.document_ids
    app_data["doc_topic"] = model.doc_topic
    app_data["topic_sizes"] = model.topic_sizes
    app_data["topic_words"] = model.topic_words
    app_data["doc_topic_facet"] = model.doc_topic_facet
    app_data["topic_facets"] = model.topic_facets
    app_data["topic_polygons_xs"] = topic_polygons_xs
    app_data["topic_polygons_ys"] = topic_polygons_ys

    # Serialize app input data
    PATH_INPUT_DATA = os.path.join(APP_DIR "input_data.pickle")
    joblib.dump(app_data, PATH_INPUT_DATA)
