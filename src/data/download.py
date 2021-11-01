import osmnx as ox
from shapely import wkt
from geopandas import GeoDataFrame
from typing import Union, Dict, List
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon
from .utils import TOP_LEVEL_OSM_TAGS

def ensure_geometry_type(
    df: GeoDataFrame, geometry_column: str = "geometry"
) -> GeoDataFrame:
    def ensure_geometry_type_correct(geometry):
        if type(geometry) == str:
            return wkt.loads(geometry)
        else:
            return geometry

    if geometry_column in df.columns:
        df[geometry_column] = df[geometry_column].apply(ensure_geometry_type_correct)
    return df


def download_whole_city(city_name: Union[str, List[str]], save_path: Path, timeout: int = 10000):
    if type(city_name) == str:
        name = city_name
    else:
        name = city_name[0]
    print(name)
    area_path = save_path.joinpath(name)
    area_path.mkdir(parents=True, exist_ok=True)
    for tag in tqdm(TOP_LEVEL_OSM_TAGS):
        tag_path = area_path.joinpath(tag + ".pkl")
        if not tag_path.exists():
            tag_gdf = download_whole_osm_tag(city_name, tag, timeout)
            if tag_gdf.empty:
                print(f"Tag: {tag} empty for city: {city_name}")
            else:
                tag_gdf.to_pickle(tag_path)
        else:
            print(f"Tag: {tag} exists for city: {city_name}")


def download_whole_osm_tag(
    area_name: Union[str, Dict[str, str]], tag: str, timeout: int = 10000
) -> GeoDataFrame:
    return download_specific_tags(area_name, {tag: True}, timeout)


def download_specific_tags(
    area_name: Union[str, Dict[str, str]],
    tags: Dict[str, Union[str, bool]],
    timeout: int = 10000,
) -> GeoDataFrame:
    ox.config(timeout=timeout)
    geometries_df = ox.geometries_from_place(area_name, tags=tags)
    geometries_df = ensure_geometry_type(geometries_df)
    return geometries_df


def get_bounding_gdf(city: str) -> GeoDataFrame:
    # query = {"city": city}
    query = city
    gdf = ox.geocode_to_gdf(query)
    return gdf


def get_bounding_polygon(city: str) -> Polygon:
    gdf = get_bounding_gdf(city)
    return gdf["geometry"][0]
