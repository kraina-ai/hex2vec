from geopandas import GeoDataFrame
from pyproj import crs
from src.data.download import get_bounding_gdf
import h3
from typing import Callable, List, Union
from shapely.geometry import Point, Polygon, geo, MultiPolygon
import pandas as pd
from shapely.geometry import mapping
from tqdm import tqdm
import geopandas as gpd
from src.data.load_data import load_city_tag
from src.data.utils import TOP_LEVEL_OSM_TAGS
from src.settings import DATA_INTERIM_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR
from pathlib import Path
from src.data.load_data import load_city_tag_h3, load_grouped_city
import numpy as np
from typing import Dict


def h3_to_polygon(hex: int) -> Polygon:
    boundary = h3.h3_to_geo_boundary(hex)
    boundary = [[y, x] for [x, y] in boundary]
    return Polygon(boundary)


def get_hexes_for_place(
    area_gdf: GeoDataFrame, resolution: int, return_gdf=False
) -> Union[List[int], GeoDataFrame]:
    original = area_gdf
    buffered = get_buffered_place_for_h3(original, resolution)
    geojson = mapping(buffered)
    all_hexes = []
    all_hex_polygons = []
    for feature in geojson["features"]:
        geom = feature["geometry"]
        geom["coordinates"] = [[[j[1], j[0]] for j in i] for i in geom["coordinates"]]
        hexes = list(h3.polyfill(geom, resolution))
        all_hexes.extend(hexes)
        hex_polygons = list(map(h3_to_polygon, hexes))
        all_hex_polygons.extend(hex_polygons)
    hexes_gdf = GeoDataFrame(
        pd.DataFrame({"h3": all_hexes, "geometry": all_hex_polygons}), crs="EPSG:4326"
    )

    intersecting_hexes_gdf = gpd.sjoin(hexes_gdf, original)
    intersecting_hexes_gdf = intersecting_hexes_gdf[["h3", "geometry"]]
    intersecting_hexes_gdf.drop_duplicates(inplace=True)
    if return_gdf:
        return intersecting_hexes_gdf
    else:
        return intersecting_hexes_gdf.h3.tolist()


def get_buffered_place_for_h3(place: GeoDataFrame, resolution: int) -> GeoDataFrame:
    twice_edge_length = 2 * h3.edge_length(resolution=resolution, unit="m")
    buffered = place.copy()
    buffered["geometry"] = (
        place.to_crs(epsg=3395).buffer(int(twice_edge_length)).to_crs(epsg=4326)
    )
    return buffered


def prepare_city_path(base_path: Path, city: str) -> Path:
    city_path = base_path.joinpath(city)
    city_path.mkdir(parents=True, exist_ok=True)
    return city_path


def get_hexes_polygons_for_city(
    city: Union[str, List[str]], resolution: int, use_cache=False
) -> GeoDataFrame:
    city_name = city if type(city) == str else city[0]
    city_path = prepare_city_path(DATA_RAW_DIR, city_name)
    cache_file = city_path.joinpath(f"h3_{resolution}.geojson")
    if use_cache and cache_file.exists() and cache_file.is_file():
        return gpd.read_file(cache_file)
    bounding_gdf = get_bounding_gdf(city)
    if type(bounding_gdf.geometry[0]) == MultiPolygon:
        bounding_gdf = bounding_gdf.explode(ignore_index=True).reset_index(drop=True)
    hexes_gdf = get_hexes_for_place(bounding_gdf, resolution, return_gdf=True)
    hexes_gdf.to_file(cache_file, driver="GeoJSON")
    return hexes_gdf


def add_h3_indices(
    gdf: GeoDataFrame, city: Union[str, List[str]], resolution: int
) -> GeoDataFrame:
    hexes_polygons_gdf = get_hexes_polygons_for_city(city, resolution)
    return gpd.sjoin(gdf, hexes_polygons_gdf, how="inner", predicate="intersects")


def add_h3_indices_to_city(
    city: Union[str, List[str]], resolution: int, data_dir: Path = None
) -> GeoDataFrame:
    city_name = city if type(city) == str else city[0]
    if data_dir is not None:
        DATA_INTERIM_DIR = data_dir.joinpath("interim")
        DATA_RAW_DIR = data_dir.joinpath("raw")
    city_destination_path = prepare_city_path(DATA_INTERIM_DIR, city_name)
    for tag in TOP_LEVEL_OSM_TAGS:
        result_path = city_destination_path.joinpath(f"{tag}_{resolution}.pkl")
        if not result_path.exists():
            print(f"Loading data for {city_name} {tag}")
            tag_gdf = load_city_tag(city_name, tag, data_dir=DATA_RAW_DIR)
            if tag_gdf is not None:
                print(f"Adding h3 indices to {city_name} {tag}")
                tag_gdf = tag_gdf
                h3_gdf = add_h3_indices(tag_gdf, city, resolution)
                h3_gdf.to_pickle(result_path)
            else:
                print(f"Tag {tag} doesn't exist for city {city}, skipping...")
        # else:
        #     print(f"{result_path} already exists, skipping...")


def merge_all_tags_for_city(city: str, resolution: int):
    city_path = prepare_city_path(DATA_INTERIM_DIR, city)
    dfs = []
    for tag in TOP_LEVEL_OSM_TAGS:
        try:
            gdf = load_city_tag(city, tag, data_dir=DATA_RAW_DIR)
            if gdf is not None:
                gdf = add_h3_indices(gdf, city, resolution)
                dfs.append(gdf)
        except FileNotFoundError:
            print(f"Tag {tag} file not found for {city}")
    city_df = pd.concat(dfs, ignore_index=True)
    city_file_path = city_path.joinpath(f"all_{resolution}.pkl")
    city_df.to_pickle(city_file_path)
    return city_df


def group_df_by_tag_values(df, tag: str):
    tags = df.reset_index(drop=True)[["h3", tag]]
    indicators = tags[[tag]].pivot(columns=tag, values=tag)
    indicators[indicators.notnull()] = 1
    indicators.fillna(0, inplace=True)
    indicators = indicators.add_prefix(f"{tag}_")
    return pd.concat([tags, indicators], axis=1).groupby("h3").sum().reset_index()


def group_city_tags(
    city: str,
    resolution: int,
    tags=TOP_LEVEL_OSM_TAGS,
    filter_values: Dict[str, str] = None,
    fill_missing=True,
) -> pd.DataFrame:
    dfs = []
    for tag in tags:
        df = load_city_tag_h3(city, tag, resolution, filter_values)
        if df is not None and not df.empty:
            tag_grouped = group_df_by_tag_values(df, tag)
        else:
            tag_grouped = pd.DataFrame()
        if fill_missing and filter_values is not None:
            columns_names = [f"{tag}_{value}" for value in filter_values[tag]]
            for c_name in columns_names:
                if c_name not in tag_grouped.columns:
                    tag_grouped[c_name] = 0
        dfs.append(tag_grouped)

    results = pd.concat(dfs, axis=0)
    results = results.fillna(0).groupby("h3").sum().reset_index()

    city_destination_path = prepare_city_path(DATA_PROCESSED_DIR, city)
    file_path = city_destination_path.joinpath(f"{resolution}.pkl")
    results.to_pickle(file_path)
    return results


def group_city_top_level_tags(
    city: str, resolution: int, tags=TOP_LEVEL_OSM_TAGS
) -> pd.DataFrame:
    dfs = []
    for tag in tags:
        df = load_city_tag_h3(city, tag, resolution)
        if df is not None:
            df = df[["h3", tag]]
            dfs.append(df)

    results = pd.concat(dfs, axis=0)
    for tag in tags:
        if tag not in results.columns:
            results[tag] = np.nan
    df_tags = results[tags]
    df_tags = df_tags.notna().astype(float)
    results[tags] = df_tags
    df_top_level = results.groupby("h3").sum().reset_index()[["h3", *tags]]
    return df_top_level


def add_geometry_to_df(df: pd.DataFrame) -> GeoDataFrame:
    df["geometry"] = df["h3"].apply(h3_to_polygon)
    gdf = GeoDataFrame(df, crs="EPSG:4326")
    return gdf


def group_cities(cities: str, resolution: int, add_city_column=True) -> pd.DataFrame:
    dfs = []
    for city in cities:
        df = load_grouped_city(city, resolution)
        print(city)
        print(df.isna().sum().sum())
        print(len(df.columns))
        if add_city_column:
            df["city"] = city
        dfs.append(df)

    all_cities = pd.concat(dfs, axis=0, ignore_index=True).set_index("h3")
    print(all_cities.isna().sum().sum())
    all_cities.to_pickle(DATA_PROCESSED_DIR.joinpath(f"{resolution}.pkl"))
    return all_cities
