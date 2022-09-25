import asyncio
import itertools
from typing import Iterable, List
import pandas as pd
from pathlib import Path
import click
import tqdm
import h3

from src.data.download import download_whole_city
from src.data.make_dataset import (
    add_h3_indices_to_city_explicit_paths,
    group_city_tags,
    h3_to_polygon,
    save_hexes_polygons_for_city,
)
from src.data.load_data import load_filter
from src.data.utils import TOP_LEVEL_OSM_TAGS
from src.settings import DATA_DIR, DATA_RAW_DIR, FILTERS_DIR
from src.utils.tesselate_amazon_data import create_city_from_hex, group_hex_tags, join_hex_dfs, pull_tags_for_hex, pull_tags_for_hex_gdf, walk_n_queue


def _check_dir_exists(dir_path: str) -> Path:
    dir_path = Path(dir_path)
    assert dir_path.is_dir() and dir_path.exists(), "Data directory does not exist"
    return dir_path


def _iter_cities(data_dir: Path) -> Iterable[Path]:
    for city in data_dir.iterdir():
        if city.is_dir():
            yield city


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--resolution", type=int, default=9, help="H3 resolution")
def group_city_hexagons(data_dir: str, output_dir: str, resolution: int):
    data_dir = _check_dir_exists(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    group_hex_tags(
        hex_parent_dir=data_dir,
        tag_list=TOP_LEVEL_OSM_TAGS,
        output_dir=data_dir,
        resolution=9,
        filter_values=load_filter(Path("/Users/max/Development/green-last-mile/hex2vec/filters/from_wiki.json")),    
    )


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("interim_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--raw-resolution", type=int)
@click.option("--resolution", "-r", multiple=True, type=int, )
def group_all_city_hexagons(data_dir: str, interim_dir: str, output_dir: str, raw_resolution: int, resolution: List[int]):
    # make sure the data directory exists
    data_dir = _check_dir_exists(data_dir)
    interim_dir = _check_dir_exists(interim_dir)

    # create the output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

 
    # loop and create dataframe. No mp for RAM issues
    for city, res in itertools.product(
            _iter_cities(data_dir), resolution
        ):

        # for city, res in itertools.product(_iter_cities(data_dir), resolutions):
        print(f"Processing {city.name} - h3 {res}")
        interim_path = interim_dir / city.stem / f"resolution_{res}"
        interim_path.mkdir(parents=True, exist_ok=True)
        
        join_hex_dfs(
            city.joinpath(f"resolution_{raw_resolution}"), 
            TOP_LEVEL_OSM_TAGS, 
            res, 
            interim_path
        )


        group_hex_tags(
            hex_parent_dir=interim_path,
            tag_list=TOP_LEVEL_OSM_TAGS,
            output_dir=interim_path,
            resolution=res,
            filter_values=load_filter(Path() / "filters" / "from_wiki.json"),    
        )

        city_out_path = output_dir / city.stem 
        city_out_path.mkdir(exist_ok=True, parents=True)

        create_city_from_hex(
            hex_parent_dir=interim_path,
            output_dir=city_out_path,
            resolution=res
        )


@click.command()
@click.argument("city")
def run():
    cities = [
        "Seattle, USA",
        "Chicago, USA",
        "Los Angeles, USA",
        "Boston, USA",
        "Austin, USA",
    ]

    for city in tqdm(cities):
        download_whole_city(city, DATA_RAW_DIR)


@click.command()
@click.argument("data-dir", type=click.Path(exists=True))
@click.option("--resolution", default=9, help="hexbin size")
def add_h3_indices(
    data_dir: Path,
    resolution: int,
):

    data_dir = _check_dir_exists(data_dir)
    interem_dir = data_dir / "intermediate"
    for city in _iter_cities(data_dir):
        if city != interem_dir:
            add_h3_indices_to_city_explicit_paths(
                city.stem, resolution, data_dir, interim_dir=interem_dir
            )


@click.command()
@click.argument("data-dir")
@click.argument("output-dir")
def simplify_data(
    data_dir: str,
    output_dir: str,
):
    """
    Simplify the data by removing all columns except for the tag and the geometry.

    Args:
        data_dir: Path to the data directory
        output_path: Path to the output directory
    """
    output_path = Path(output_dir)
    data_dir = _check_dir_exists(data_dir)

    # make the output directory if it doesn't exist
    if not output_path.exists():
        output_path.mkdir()

    for city_path in _iter_cities(data_dir):
        output_city = output_path.joinpath(city_path.stem)
        output_city.mkdir(exist_ok=True)
        for tag in city_path.glob("*.pkl"):
            print(tag)
            df = pd.read_pickle(tag)
            assert (
                tag.stem in df.columns
            ), f"Tag {tag.stem} not in dataframe for {city_path.stem}"
            df = df[["geometry", tag.stem]]
            df.to_pickle(output_city.joinpath(f"{tag.stem}.pkl"), protocol=4)


@click.command()
@click.argument("data-dir")
@click.argument("output-dir")
@click.option("--resolution", default=9, help="hexbin size")
@click.option("--filter-file", default=lambda: FILTERS_DIR.joinpath("from_wiki.json"), help="Path to json file with key value pairs to filter on")
def group_all_city_tags(data_dir: str, output_dir: str, resolution: int, filter_file: str) -> None:
    """
    Group all tags for each city into a single dataframe.

    data_dir (str): Path to the data directory\\n
    output_dir (str): Path to the output directory
    """
    data_dir = _check_dir_exists(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    filter_tags = load_filter(Path(filter_file)) if filter_file else None
    for city in _iter_cities(data_dir):
        print(f"Processing {city.stem}")
        group_city_tags(city.stem, resolution, filter_values=filter_tags, data_dir=data_dir, save_dir=output_path)



async def _pull_hex_gdf(latlon_df: pd.DataFrame, data_dir: Path, tag_list: List[str], level: int) -> None:
        
    # create a queue of needed files
    q = asyncio.Queue()
    s = asyncio.Semaphore(100)
    tasks = []
    for c, df in latlon_df.groupby('city'):
        city_dir = data_dir.joinpath(c)
        city_dir.mkdir(parents=True, exist_ok=True)
        tasks.append(walk_n_queue(q, city_dir, df, tag_list, level))
    
    
    # consume the queue asyncronously
    consumers = []
    for _ in range(100):
        consumers.append(asyncio.create_task(
            pull_tags_for_hex(
                q,
                s,

            )
        ))
    
    await asyncio.gather(*tasks)
    await q.join()

    # kill the consumers now that the task has been finished
    for c in consumers:
        c.cancel()


@click.command()
@click.argument("output-dir")
@click.option("--latlon-csv", type=click.Path(exists=True), help="path to file lat/lng pairs. Columns should be 'lat', 'lng', and 'city'")
@click.option("--city", "-c", multiple=True, type=str)
@click.option("--level", "-l", type=int, help="H3 level for which to pull")
def pull_all(output_dir, latlon_csv, city, level) -> None:
    # necessary imports
    import osmnx as ox
    from shapely.geometry import mapping
    
    # settings
    # ox.settings.overpass_rate_limit = False
    # ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api"
    ox.settings.timeout = 10_000

    # handle paths
    output_path = Path(output_dir)

    if latlon_csv:
        # first find the hexagons that tessalate the csv and seperate out the 
        latlon = pd.read_csv(latlon_csv)
        # add in the h3 index
        latlon['h3'] = latlon[['lat', 'lng']].apply(lambda x: h3.geo_to_h3(x[0], x[1], level), raw=True, axis=1)
        
        # drop individual lat/lons, just keep unique h3s, cities, and geometries
        latlon = latlon.groupby('h3').first().reset_index()
        
        # buffer by one neighbor. 
        dfs = []
        for city, group_df in latlon.groupby('city'):
            h3s = set(group_df['h3'].values)
            new_rows = []
            reported_hex = set()
            for hex in h3s:
                # find the missing
                if missing := h3.k_ring(hex, 1) - h3s:
                    for _h in missing:
                        if _h not in reported_hex:
                            new_rows.append({'city': city, 'h3': _h, 'lat': None, 'lng': None, } )
                            reported_hex.add(
                                _h
                            )
            dfs.append(
                pd.concat(
                    [group_df, pd.DataFrame.from_records(new_rows)],
                    axis=0
                ).reset_index()
            )

        latlon = pd.concat(dfs).reset_index()
            
        # add in an additional buffer of h3s
        latlon['geometry'] = list(map(h3_to_polygon, latlon['h3']))
        

        # pull all of the required hexagons asynchronously
        asyncio.run(
            _pull_hex_gdf(latlon_df=latlon, data_dir=output_path, tag_list=TOP_LEVEL_OSM_TAGS, level=level)
        )
    
    if city:
        # pull cities that may be desired

        dfs = []
        for c in city:
            # find the boundary polygon of the city
            city_df = ox.geometries_from_place(
                c, tags={"boundary": "administrative"}
            )
            city_boundary = city_df.loc[city_df["name"] == c.split(",")[0], "geometry"].iloc[0]

            # cover the polygon with hexes
            city_boundary_geojson = mapping(city_boundary)
            city_boundary_geojson['coordinates'] = [[c[::-1] for c in coords] for coords in city_boundary_geojson['coordinates']]
            desired_h3s = list(h3.polyfill(city_boundary_geojson, level))

            # create a dataframe representing the hexagons
            city_df = pd.DataFrame({
                'h3': desired_h3s,
                'geometry': list(map(h3_to_polygon, desired_h3s))
            })

            city_df['city'] = c

            dfs.append(
                city_df.copy()
            )

        asyncio.run(
            _pull_hex_gdf(latlon_df=pd.concat(dfs, axis=0), data_dir=output_path, tag_list=TOP_LEVEL_OSM_TAGS, level=level)
        )
        
    return output_path



@click.group()
def main():
    """
    Script to automate the download and processing of the data.

    Added by Green Last Mile
    """
    pass

main.add_command(pull_all)
main.add_command(add_h3_indices)
main.add_command(simplify_data)
# main.add_command(create_h3_indices)
main.add_command(group_all_city_tags)
main.add_command(group_all_city_hexagons)
if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(e)
        exit(1)
