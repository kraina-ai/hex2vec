from audioop import add
import os
import h3
from geopandas import GeoDataFrame
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from pandas.core.frame import DataFrame
from src.settings import DATA_INTERIM_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, FILTERS_DIR
from pathlib import Path
from typing import Dict, Generator, List
import json


# def split_n_save_df(df: DataFrame, path: Path, n: int) -> None:
#     base_name = path.name[:path.name.rfind(".")]
#     chunk_size = int(len(df) // n)  # force cast to int
#     print(len(df))
#     i = 0  # because we will return the first chunk
#     while len(df) > chunk_size:
#         base_name_i = f"{base_name}_{i}.pkl"
#         # df.iloc[chunk_size * i : (chunk_size + 1) * i].to_pickle(path.parent.joinpath(base_name_i), protocol=4)
#         _len = len(df)
#         df.iloc[_len - chunk_size:].to_pickle(path.parent.joinpath(base_name_i), protocol=4)
#         df = df.iloc[:_len - chunk_size]
#         i += 1
#     return df

# def read_chunk_df(path: Path, ) -> Generator[DataFrame, None, None]:
#     n = 0
#     while True:
#         base_name = f'{path.name[:path.name.rfind(".")]}_{n}.pkl'
#         if not path.parent.joinpath(base_name).exists():
#             break
#         df = pd.read_pickle(path.parent.joinpath(base_name), )
#         os.remove(path.parent.joinpath(base_name))
#         n += 1
#         yield df


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
    df = pd.read_pickle(path)
    return GeoDataFrame(df, crs="EPSG:4326")


def load_city_tag(
    city: str,
    tag: str,
    split_values=True,
    filter_values: Dict[str, str] = None,
    data_dir: Path = DATA_RAW_DIR,
) -> GeoDataFrame:
    path = data_dir.joinpath(city, f"{tag}.pkl")
    if path.exists():
        gdf = []
        for df in split_big_df(path, keep_columns=["osmid", tag, "geometry"]):
            _gdf = GeoDataFrame(df, crs="EPSG:4326")
            if split_values:
                # split values into separate columns
                _gdf[tag] = _gdf[tag].str.split(";")
                _gdf = _gdf.explode(tag)
                _gdf[tag] = _gdf[tag].str.strip()
            _gdf = filter_gdf(_gdf, tag, filter_values)
            gdf.append(_gdf)
        return pd.concat(gdf, sort=False)
    else:
        return None


def load_city_tag_h3(
    city: str,
    tag: str,
    resolution: int,
    filter_values: Dict[str, str] = None,
    data_path: Path = DATA_INTERIM_DIR,
) -> GeoDataFrame:
    path = data_path.joinpath(city, f"{tag}_{resolution}.pkl")
    if path.exists():
        gdf = load_gdf(path)
        gdf[tag] = gdf[tag].str.split(";")
        gdf = gdf.explode(tag)
        gdf[tag] = gdf[tag].str.strip()
        gdf = filter_gdf(gdf, tag, filter_values)
        return gdf
    else:
        return None


def filter_gdf(
    gdf: GeoDataFrame, tag: str, filter_values: Dict[str, str]
) -> GeoDataFrame:
    if filter_values is not None:
        selected_tag_values = set(filter_values[tag])
        gdf = gdf[gdf[tag].isin(selected_tag_values)]
    return gdf


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
    dataset_path = data_dir.joinpath(f"{resolution}.pkl") if file_path is None else file_path
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
