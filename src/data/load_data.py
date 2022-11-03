from audioop import add
import os
import h3
import geopandas as gpd
from geopandas import GeoDataFrame
import geopolars as gp
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from pandas.core.frame import DataFrame
from src.settings import DATA_INTERIM_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, FILTERS_DIR
from pathlib import Path
from typing import Dict, Generator, List
import json5 as json

from src.utils.advanced_tags import Tag


def split_big_df(
    path: Path, keep_columns: List[str]
) -> Generator[DataFrame, None, None]:
    df = pd.read_pickle(path).reset_index()[keep_columns]
    gb = df.memory_usage(deep=True).sum() / 1000000000.0
    # print("Size of df:", gb)
    if gb > 3:
        print("[!] Warning - df is potentially too big to load into memory")
    yield df


def load_gdf(path: Path, crs="EPSG:4326") -> GeoDataFrame:
    if ".feather" in path.suffixes:
        df = None
        for method in [gpd, pd]:
            try:
                df = method.read_feather(path)
                break
            except Exception as e:
                continue
        assert df is not None
    else:
        df = pd.read_pickle(path)

    if "geometry" in df.columns:
        return GeoDataFrame(df, crs=crs)
    else:
        return GeoDataFrame(df)


def load_city_tag(
    city: str,
    tag: Tag,
    split_values=True,
    data_dir: Path = DATA_RAW_DIR,
) -> GeoDataFrame:
    path = data_dir.joinpath(city, f"{tag.osmxtag}.pkl")
    if path.exists():
        gdf = []
        for df in split_big_df(path, keep_columns=["osmid", tag.osmxtag, "geometry"]):
            _gdf = GeoDataFrame(df, crs="EPSG:4326")
            if split_values:
                # split values into separate columns
                _gdf[str(tag.osmxtag)] = _gdf[tag.osmxtag].str.split(";")
                _gdf = _gdf.explode(tag.osmxtag)
                _gdf[tag.osmxtag] = _gdf[tag.osmxtag].str.strip()
                _gdf = tag.filter_df_by_tag_values(_gdf)
            gdf.append(_gdf)
        return pd.concat(gdf, sort=False)
    else:
        return None


def load_city_tag_h3(
    city: str,
    tag: Tag,
    resolution: int,
    data_path: Path = DATA_INTERIM_DIR,
) -> GeoDataFrame:
    path = data_path.joinpath(city, tag.file_name(f"_{resolution}", ".feather"))
    if path.exists():
        return load_gdf(path)
    else:
        return None


def load_filter(
    filter_file_path: Path, values_to_drop: Dict[str, List[str]] = None
) -> Dict[str, List[str]]:
    if filter_file_path.exists() and filter_file_path.is_file():
        with filter_file_path.open(mode="rt") as f:
            filter_values = json.load(f)
            if values_to_drop is not None:
                for key, values in values_to_drop.items():
                    for value in values:
                        filter_values[key].remove(value)
            return filter_values
    else:
        available_filters = [f.name for f in FILTERS_DIR.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Filter {filter_file_path.stem} not found. Available filters: {available_filters}"
        )


def load_grouped_city(city: str, resolution: int) -> DataFrame:
    city_df_path = DATA_PROCESSED_DIR.joinpath(city).joinpath(f"{resolution}.pkl")
    return pd.read_pickle(city_df_path)


def load_processed_dataset(
    resolution: int,
    select_cities: List[str] = None,
    drop_cities: List[str] = None,
    select_tags: List[str] = None,
    data_dir: Path = DATA_PROCESSED_DIR,
    file_path: Path = None,
    add_city_column: bool = False,
    city_column_name: str = "city",
    city: str = None,
) -> DataFrame:
    dataset_path = (
        data_dir.joinpath(f"{resolution}.pkl") if file_path is None else file_path
    )
    df = pd.read_pickle(dataset_path)
    if add_city_column:
        df[city_column_name] = city
    if select_cities is not None:
        df = df[df["city"].isin(select_cities)]
    if drop_cities is not None:
        df = df[~df["city"].isin(drop_cities)]
    if select_tags is not None:
        df = df[[*df.columns[df.columns.str.startswith(tuple(select_tags))], "city"]]
    df = df[~(df.drop(columns="city") == 0).all(axis=1)]
    return df
