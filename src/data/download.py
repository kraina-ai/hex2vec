import asyncio
from functools import wraps, partial
import osmnx as ox
from shapely import wkt
from geopandas import GeoDataFrame
from typing import Union, Dict, List
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon
from .utils import TOP_LEVEL_OSM_TAGS
from shapely.errors import ShapelyDeprecationWarning
import warnings

def ensure_geometry_type(df: GeoDataFrame, geometry_column: str = "geometry") -> GeoDataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
        def ensure_geometry_type_correct(geometry):
            return wkt.loads(geometry) if type(geometry) == str else geometry

        if geometry_column in df.columns:
            df[geometry_column] = df[geometry_column].apply(ensure_geometry_type_correct)
        return df


def download_whole_city(city_name: Union[str, List[str]], save_path: Path, timeout: int = 10000):
    name = city_name if type(city_name) == str else city_name[0]
    print(name)
    area_path = save_path.joinpath(name)
    area_path.mkdir(parents=True, exist_ok=True)
    for tag in tqdm(TOP_LEVEL_OSM_TAGS):
        tag_path = area_path.joinpath(f"{tag}.pkl")
        if not tag_path.exists():
            tag_gdf = download_whole_osm_tag(city_name, tag, timeout)
            if tag_gdf.empty:
                print(f"Tag: {tag} empty for city: {city_name}")
            else:
                tag_gdf.to_pickle(tag_path)
        else:
            print(f"Tag: {tag} exists for city: {city_name}")


# little of a hack to get around the fact that osmnx doesn't have async
def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return run

@async_wrap
def download_whole_city_async(*args, **kwargs):
    download_whole_city(*args, **kwargs)



def download_whole_osm_tag(
    area_name: Union[str, Dict[str, str]], tag: str, timeout: int = 10000
) -> GeoDataFrame:
    return download_specific_tags(area_name, {tag: True}, timeout)


def download_specific_tags(
    area_name: Union[str, Dict[str, str]],
    tags: Dict[str, Union[str, bool]],
    timeout: int = 10000,
) -> GeoDataFrame:
    ox.settings.timeout = timeout
    geometries_df = ox.geometries_from_place(area_name, tags=tags)
    geometries_df = ensure_geometry_type(geometries_df)
    return geometries_df


def get_bounding_gdf(city: str) -> GeoDataFrame:
    query = city
    return ox.geocode_to_gdf(query)


def get_bounding_polygon(city: str) -> Polygon:
    gdf = get_bounding_gdf(city)
    return gdf["geometry"][0]
