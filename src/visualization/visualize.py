from pathlib import Path
from shutil import which
from geopandas.geodataframe import GeoDataFrame
from ipywidgets.widgets import widget
from keplergl import KeplerGl
import pandas as pd
import geopandas as gpd
from typing import Union
from .config import load_config
import contextily as ctx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from src.data.make_dataset import h3_to_polygon
from src.settings import FIGURES_DIR
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from src.data.download import ensure_geometry_type


def visualize_kepler(
    data: Union[pd.DataFrame, gpd.GeoDataFrame], name="data", config_name: str = None
) -> KeplerGl:
    if config_name is not None:
        config = load_config(config_name)
        if config is not None:
            return KeplerGl(data={name: data}, config=config, height=900)
    return KeplerGl(data={name: data})


def visualize_clusters_kepler(
    data: Union[pd.DataFrame, gpd.GeoDataFrame], name="data"
) -> KeplerGl:
    return visualize_kepler(data, name=name, config_name="clusters")


def visualize_df(
    df: Union[pd.DataFrame, GeoDataFrame],
    map_source=ctx.providers.CartoDB.Positron,
    column="label",
    alpha=0.6,
    figsize=(15, 15),
    **kwargs,
):
    if type(df) == pd.DataFrame or "geometry" not in df.columns:
        if "h3" in df.columns:
            df = df.copy(deep=False)
            df["geometry"] = df["h3"].apply(h3_to_polygon)
            df = gpd.GeoDataFrame(df, crs="EPSG:4326")
        else:
            raise ValueError(
                "Passed dataframe must either be GeoDataFrame with geometry column or have h3 column"
            )
    ax = df.to_crs(epsg=df["geometry"].estimate_utm_crs()).plot(
        column=column, legend=True, alpha=alpha, figsize=figsize, **kwargs
    )
    ctx.add_basemap(ax, source=map_source, attribution_size=4)
    ax.axis("off")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        labelbottom=False,
        top=False,
        labeltop=False,
        left=False,
        labelleft=False,
        right=False,
        labelright=False,
    )
    return ax.get_figure(), ax


def visualize_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def save_kepler_map(kepler_map: KeplerGl, figure_subpath: Path, remove_html=False):
    result_path = FIGURES_DIR.joinpath(figure_subpath)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    html_file = result_path.with_suffix(".html")

    for gdf in kepler_map.data.values():
        ensure_geometry_type(gdf)
    kepler_map.save_to_html(file_name=html_file)

    options = Options()
    height = kepler_map.height
    width = 1300
    options.add_argument("--headless")
    options.add_argument(f"--window-size={width},{height}")

    driver = webdriver.Chrome(options=options)
    driver.get(str(html_file.resolve()))
    time.sleep(3)
    driver.save_screenshot(str(result_path))
    if remove_html:
        html_file.unlink()
