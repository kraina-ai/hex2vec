import itertools
from typing import Iterable
import pandas as pd
from pathlib import Path
import click
import tqdm

from src.data.download import download_whole_city
from src.data.make_dataset import (
    add_h3_indices_to_city_explicit_paths,
    group_city_tags,
    save_hexes_polygons_for_city,
)
from src.data.load_data import load_filter
from src.data.utils import TOP_LEVEL_OSM_TAGS
from src.settings import DATA_DIR, DATA_RAW_DIR, FILTERS_DIR
from src.utils.tesselate_amazon_data import group_hex_tags


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
@click.option("--resolution", type=int, default=9, help="H3 resolution")
@click.option(
    "--sweep-resolutions",
    type=str,
    default="",
    help="Comma-separated list of resolutions to sweep",
)
def create_h3_indices(
    data_dir: Path, resolution: int, sweep_resolutions: str, *args, **kwargs
):
    """
    Create h3 indices with specified resolution for all cities in the data directory.
    """
    try:
        import os
        from joblib import Parallel, delayed

        parallel = True
    except ImportError:
        print("joblib not installed, skipping parallel processing")
        parallel = False
    if sweep_resolutions:
        resolutions = [int(r) for r in sweep_resolutions.split(",")]
    else:
        resolutions = [resolution]
    data_dir = _check_dir_exists(data_dir)
    if parallel:
        Parallel(n_jobs=os.cpu_count() - 2)(
            delayed(save_hexes_polygons_for_city)(city, resolution)
            for city, resolution in itertools.product(
                _iter_cities(data_dir), resolutions
            )
        )

    else:
        for city, resolution in itertools.product(_iter_cities(data_dir), resolutions):
            print(f"Processing {city.name} - h3 {resolution}")
            save_hexes_polygons_for_city(city, resolution)


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


@click.command()
@click.argument("data-dir")
@click.argument("output-dir")
@click.option("--resolution", default=9, help="hexbin size")
@click.option("--filter-file", default=lambda: FILTERS_DIR.joinpath("from_wiki.json"), help="Path to json file with key value pairs to filter on")
def tesselate_and_pull_city(data_dir: str, output_dir: str, resolution: int, filter_file: str) -> None:
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



@click.group()
def main():
    """
    Script to automate the download and processing of the data.

    Added by Green Last Mile
    """
    pass


main.add_command(add_h3_indices)
main.add_command(simplify_data)
main.add_command(create_h3_indices)
main.add_command(group_all_city_tags)
main.add_command(group_city_hexagons)

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(e)
        exit(1)
