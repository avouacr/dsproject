""""""
import os
from os.path import abspath, dirname
import joblib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from bokeh.layouts import row, column, layout
from bokeh.models import (
    ColumnDataSource, HoverTool, Range1d, Title, TableColumn, DataTable,
    StringFormatter, CustomJS, Div, Button
    )
from bokeh.models.widgets import Tabs, Panel
from bokeh.plotting import figure, curdoc


def generate_background(path_save):
    """"""
    # Set graph ranges
    x_coords = app_input_data["umap_2d_embeddings"][:, 0]
    y_coords = app_input_data["umap_2d_embeddings"][:, 1]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_margin_l, x_margin_r = MARGIN_L*(x_max - x_min), MARGIN_R*(x_max - x_min)
    y_margin_t, y_margin_b = MARGIN_T*(x_max - x_min), MARGIN_B*(x_max - x_min)

    # Generate background image (scatter plot of all documents)
    px_ratio = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1900*px_ratio, 1900*px_ratio))
    ax.scatter(x_coords, y_coords, alpha=0.1, s=1, color="grey",
               # color=[to_hex(palette[t]) for t in app_input_data["doc_topic"]]
               )
    ax.set_xlim([x_min-x_margin_l, x_max+x_margin_r])
    ax.set_ylim([y_min-y_margin_b, y_max+y_margin_t])
    ax.set_facecolor("#2F2F2F")
    ax.axes.set_axis_off()
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1)

    # Save figure
    dir_save = os.path.dirname(path_save)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    fig.savefig(path_save, bbox_inches='tight', pad_inches=0)


def map_themes(themes_dict):
    """"""
    topics = list(np.unique(app_input_data["doc_topic"]))
    topics_themes = [themes_dict[theme] for theme in themes_dict.keys()]
    topics_themes = sorted([x for sublist in topics_themes for x in sublist])

    # Validate mapping from theme to topics
    if topics != topics_themes:
        raise ValueError("Invalid mapping from theme to topics provided.")

    # Return mapping from topic to theme
    topics_to_theme = {}
    for key, values in themes_dict.items():
        for val in values:
            topics_to_theme[val] = key
    mapping = [topics_to_theme[topic] for topic in topics]
    return mapping


def build_compute_topic_map(theme_to_topics=None):
    """"""
    # Create interactive figure
    x_coords = app_input_data["umap_2d_embeddings"][:, 0]
    y_coords = app_input_data["umap_2d_embeddings"][:, 1]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_margin_l, x_margin_r = MARGIN_L*(x_max - x_min), MARGIN_R*(x_max - x_min)
    y_margin_t, y_margin_b = MARGIN_T*(x_max - x_min), MARGIN_B*(x_max - x_min)
    plot = figure(
        sizing_mode="scale_width",
        x_range=Range1d(start=x_min-x_margin_l, end=x_max+x_margin_r),
        y_range=Range1d(start=y_min-y_margin_b, end=y_max+y_margin_t),
        x_axis_location=None, y_axis_location=None,
        tools="pan,wheel_zoom,tap", toolbar_location=None,
        active_drag="pan", active_scroll="wheel_zoom"
    )
    plot.grid.grid_line_color = None
    plot.add_layout(Title(text=" "), 'above')
    plot.add_layout(Title(text="Topic map", text_font_size="200%"), 'above')

    # Set background image (scatter plot of all documents)
    # (using bokeh's Scatter slows down the app excessively)
    plot.image_url(url=[BG_URL],
                   x=x_min-x_margin_l, y=y_min-y_margin_b,
                   w=x_max-x_min+x_margin_l+x_margin_r,
                   h=y_max-y_min+y_margin_b+y_margin_t,
                   anchor="bottom_left")

    # Color topics
    if theme_to_topics is None:
        n_topics = len(np.unique(app_input_data["doc_topic"]))
        palette = plt.get_cmap(PALETTE)(np.linspace(0, 1, n_topics))
        t_colors = ["#FFDECB" for t in np.unique(app_input_data["doc_topic"])]
        topic_to_theme = t_colors   # Hack, legend will be hidden anyway
    else:
        topic_to_theme = map_themes(theme_to_topics)
        themes = np.unique(topic_to_theme)
        n_themes = len(themes)
        themes_to_int = {theme: i for i, theme in enumerate(themes)}
        palette = plt.get_cmap(PALETTE)(np.linspace(0, 1, n_themes))
        t_colors = [to_hex(palette[themes_to_int[theme]])
                    for theme in topic_to_theme]

    # Plot topic glyphs
    topic_glyphs.data = dict(
        x=app_input_data["topic_polygons_xs"],
        y=app_input_data["topic_polygons_ys"],
        topic=list(np.unique(app_input_data["doc_topic"])),
        theme=topic_to_theme,
        color=t_colors,
        words=[" ".join(list(w)[:N_WORDS_TM_TOOLTIPS])
               for w in app_input_data["topic_words"]],
        size=np.sqrt(app_input_data["topic_sizes"]) / 1.3,
        size_str=[f"{s} documents" for s in app_input_data["topic_sizes"]]
        )

    plot.multi_polygons(name="topics", source=topic_glyphs, xs="x", ys="y",
                        color="color", fill_alpha=0.5, line_alpha=0.3,
                        hover_color="white", hover_alpha=1,
                        selection_fill_alpha=0.7, selection_line_color="white",
                        selection_line_width=2, selection_line_alpha=0.6,
                        nonselection_fill_alpha=0.5, nonselection_line_alpha=0.3,
                        legend_field="theme")
    plot.legend.background_fill_alpha = 0.8
    if theme_to_topics is None:
        plot.legend.visible = False

    # Topic hovering
    hover = HoverTool(names=["topics"])
    hover.tooltips = """
    <font size="+1">
    <strong>Topic</strong>: @topic <br>
    <strong>Size</strong>: @size_str <br>
    <strong>Words</strong>: @words
    </font>
    """
    hover.point_policy = "follow_mouse"
    plot.add_tools(hover)

    return plot, t_colors


def build_facet_map():
    """"""
    # Build base plot
    plot = figure(
        id="facet_plot", sizing_mode="scale_width",
        x_axis_location=None, y_axis_location=None,
        tools="pan,wheel_zoom,tap", toolbar_location=None,
        active_drag="pan", active_scroll="wheel_zoom"
    )
    plot.grid.grid_line_color = None

    # Titles
    plot.add_layout(Title(text=" ", text_font_style="italic", text_font_size="100%"),
                    'above')
    plot.add_layout(Title(text="Facet Map", text_font_size="200%"), 'above')

    # Glyphs
    plot.circle(name="facets", source=facet_glyphs, x="x", y="y",
                size="size", color="color", fill_alpha=0.5, line_alpha=0.3,
                hover_color="white", hover_alpha=1,
                selection_fill_alpha=0.7, selection_line_color="white",
                selection_line_width=2, selection_line_alpha=0.6,
                nonselection_fill_alpha=0.5, nonselection_line_alpha=0.3)

    # Hovering tool
    hover = HoverTool(names=["facets"])
    hover.tooltips = """
    <font size="+1">
    <strong>Facet</strong>: @facet <br>
    <strong>Words</strong>: @words <br>
    <strong>Size</strong>: @size_str <br>
    </font>
    """
    hover.point_policy = "follow_mouse"
    plot.add_tools(hover)

    return plot


def compute_facet_map(topic):
    """"""
    # Extract facets data
    t_facets = list(app_input_data["topic_facets"][topic].keys())
    t_facets_words = [' '.join(app_input_data["topic_facets"][topic][f]["words"][:N_WORDS_FM_TOOLTIPS])
                      for f in t_facets]
    t_facets_size = np.array([app_input_data["topic_facets"][topic][f]["size"]
                              for f in t_facets])

    # Repesent facets as circles centered on their 2D barycentres
    facets_barycentres = []
    for facet in t_facets:
        doc_idxs_facets = app_input_data["topic_facets"][topic][facet]["doc_idxs"]
        f_bary = np.mean(app_input_data["umap_2d_embeddings"][doc_idxs_facets], axis=0)
        facets_barycentres.append(f_bary)
    bary_2d = np.array(facets_barycentres)

    facet_data = dict(
        x=bary_2d[:, 0],
        y=bary_2d[:, 1],
        facet=t_facets,
        color=[TOPIC_COLORS[topic] for n in t_facets],
        words=t_facets_words,
        size=np.sqrt(t_facets_size)*3,
        size_str=[f"{s} documents" for s in t_facets_size])

    return facet_data


def update_facet_map(attr, old, new):
    """"""
    if new:
        # Actions when selecting a topic
        topic = int(new[0])
        facet_glyphs.data = compute_facet_map(topic=topic)
        topic_words_str = " ".join(app_input_data["topic_words"][topic][:N_WORDS_FM_TITLE])
        facet_map.above[0].text = f"Topic {topic}: {topic_words_str}"
    else:
        # Actions when clicking in the background
        facet_glyphs.data = dict(
            x=[], y=[], facet=[], color=[], words=[], size=[], size_str=[]
            )
        facet_rows.data = dict(doc_id=[], doc_text=[], sim_to_facet=[])
        facet_map.above[0].text = " "

    if facet_glyphs.selected.indices:
        # When switching topics, deselect any selected facet
        facet_glyphs.selected.indices = []


def compute_facet_table(topic=None, facet=None):
    """"""
    # Retrieve facet documents
    docs_idxs = app_input_data["topic_facets"][topic][facet]["doc_idxs"]
    docs = app_input_data["documents"][docs_idxs]
    doc_ids = app_input_data["document_ids"][docs_idxs]

    # Sort them by decreasing similarity to facet vector
    sims = np.array([app_input_data["doc_topic_facet"][idx]["facet_similarity"]
                     for idx in docs_idxs])
    idxs_sorted_sims = np.flip(np.argsort(sims))
    docs_sorted = docs[idxs_sorted_sims]
    sims_sorted = np.round(sims[idxs_sorted_sims], 2)
    doc_ids_sorted = doc_ids[idxs_sorted_sims]

    # Build data table
    rows = dict(
        doc_id=doc_ids_sorted,
        doc_text=docs_sorted,
        sim_to_facet=sims_sorted
        )

    return rows


def update_facet_table(attr, old, new):
    """"""
    if new:
        # Actions when selecting a facet
        topic = topic_glyphs.selected.indices[0]
        facet = int(new[0])
        facet_rows.data = compute_facet_table(topic=topic,
                                              facet=facet)
        facet_words_str = " ".join(app_input_data["topic_facets"][topic][facet]["words"][:N_WORDS_FT_TITLE])
        facet_table_subtitle.text = f"Facet {facet}: {facet_words_str}"
    else:
        # Actions when clicking in the background
        facet_rows.data = dict(doc_id=[], doc_text=[], sim_to_facet=[])
        facet_table_subtitle.text = " "


# Import input data
APP_DIR = dirname(abspath(__file__))
PATH_INPUT_DATA = os.path.join(APP_DIR, "input_data.pickle")
app_input_data = joblib.load(PATH_INPUT_DATA)

# App parameters
curdoc().title = "r/changemyview explorer"
PALETTE = "tab20"
MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.025, 0.15, 0.025, 0.05
CSS_TEXT_STYLE = {'width': '800px',
                  'font-family': 'Helvetica',
                  'font-size': '100%',
                  'text-align': 'justify',
                  'text-justify': 'inter-word'}

# Generate background image for app's topic map (scatter of all documents)
# (using bokeh's Scatter slows down the app excessively)
BG_URL = os.path.join(os.path.basename(os.path.dirname(__file__)),
                      "static", "points.png")
generate_background(path_save=BG_URL)

# Define themes mapping to topics (optional)
theme_dict = {
    "Sexuality": [2, 6, 11, 16, 31, 32, 35],
    "Racism": [1, 9],
    "Media": [5, 14, 40],
    "Politics": [7, 8, 13, 18, 19, 20, 22, 27, 29, 43, 45],
    "Beliefs": [3, 12, 23, 24],
    "Economy": [10, 17, 30],
    "Health": [21, 34, 37, 38, 39, 42],
    "Death": [25, 41],
    "Nature": [15, 36],
    "Education": [4],
    "Culture": [0, 26, 33],
    "Other": [28, 44]
}

# Panel 1 : app

app_title = Div(text="r/changemyview explorer", style={'font-family': 'Helvetica',
                                                       'font-size': '300%'})

# Topic map
N_WORDS_TM_TOOLTIPS = 15
topic_glyphs = ColumnDataSource(data=dict(
    x=[], y=[], topic=[], theme=[], color=[], words=[], size_str=[]
    ))
topic_map, TOPIC_COLORS = build_compute_topic_map(theme_to_topics=theme_dict)

# Facet map
N_WORDS_FM_TOOLTIPS = 15
N_WORDS_FM_TITLE = 12
facet_glyphs = ColumnDataSource(data=dict(
    x=[], y=[], facet=[], color=[], words=[], size=[], size_str=[]
    ))
facet_map = build_facet_map()

# Callback : update facet map when clicking on a topic
topic_glyphs.selected.on_change('indices', update_facet_map)

# Callback : reset zoom in facet map when selecting another topic
zoom_reset = CustomJS(code="Bokeh.documents[0].get_model_by_id('facet_plot').reset.emit()")
facet_glyphs.js_on_change('data', zoom_reset)

# Facet table
N_WORDS_FT_TITLE = 20
facet_rows = ColumnDataSource(data=dict(
    doc_id=[], doc_text=[], sim_to_facet=[]
    ))
formatter = StringFormatter(text_color="black")
columns = [
    TableColumn(field="doc_id", title="ID", width=100, formatter=formatter),
    TableColumn(field="doc_text", title="Content", width=900, formatter=formatter),
    TableColumn(field="sim_to_facet", title="Relevance", width=150, formatter=formatter)
    ]
facet_table = DataTable(source=facet_rows, columns=columns, width=1100,
                        fit_columns=False)
facet_table_title = Div(text="Facet Table", style={'font-family': 'Helvetica',
                                                   'font-size': '200%'})
facet_table_subtitle = Div(text=" ", style={'font-family': 'helvetica',
                                            'font_style': "italic",
                                            'font-size': '100%'})

# Callback : update table with facet documents when clicking on a facet
facet_glyphs.selected.on_change('indices', update_facet_table)

# Callback : download table data as csv
# dl_button = Button(label="Download", button_type="success")
# js_file_path = os.path.join(dirname(__file__), "download.js")
# dl_button.js_on_click(CustomJS(args=dict(source=facet_rows),
#                                code=open(js_file_path).read()))

# Parametrize layout
layout_main = layout(app_title,
                     column(row(topic_map, facet_map),
                            column(facet_table_title,
                                   facet_table_subtitle,
                                   facet_table),
                            sizing_mode="scale_width"))

# Panel 2 : documentation
documentation_title = Div(text="Documentation", style={'font-family': 'Helvetica',
                                                       'font-size': '300%'})

with open(os.path.join(APP_DIR, "text_documentation.html"), "r") as file:
    DOCUMENTATION_TEXT = file.read()
documentation_text_div = Div(text=DOCUMENTATION_TEXT, style=CSS_TEXT_STYLE)
layout_documentation = layout([documentation_title, documentation_text_div])

# Panel 3 : about

about_title = Div(text="About", style={'font-family': 'Helvetica',
                                       'font-size': '300%'})
with open(os.path.join(APP_DIR, "text_about.html"), "r") as file:
    ABOUT_TEXT = file.read()
about_text_div = Div(text=ABOUT_TEXT, style=CSS_TEXT_STYLE)
layout_about = layout(column(about_title, about_text_div))

# General layout of the app
tab_main = Panel(child=layout_main, title="App")
tab_documentation = Panel(child=layout_documentation, title="Documentation")
tab_about = Panel(child=layout_about, title="About")
tabs = Tabs(tabs=[tab_main, tab_documentation, tab_about])
curdoc().add_root(tabs)
