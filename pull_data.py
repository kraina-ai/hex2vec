from email.policy import default
from pathlib import Path
import click

from src.data.download import download_whole_city
from src.data.make_dataset import add_h3_indices_to_city
from src.settings import DATA_DIR, DATA_RAW_DIR




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
@click.option('--data-dir', default=DATA_DIR, help='Path to the data directory')
@click.option('--resolution', default=9, help='hexbin size')
def add_h3_indices(
    data_dir: Path,
    resolution: int,
):
    cities = [
        "Seattle, USA",
        "Chicago, USA",
        "Los Angeles, USA",
        "Boston, USA",
        "Austin, USA",
    ]

    data_dir = Path(data_dir)
    assert data_dir.exists() and data_dir.is_dir()

    for city in cities:
        add_h3_indices_to_city(city, resolution, data_dir)


# # @click.group()
# @click.command()
# @click.option('--pull-data', is_flag=True, help='Pull data from the internet', default=False)
# @click.option('--add-h3-indices', is_flag=True, help='Add h3 indices to the data', default=False)
# def main(pull_data: bool, add_h3_indices: bool):
#     assert pull_data or add_h3_indices, "You must specify at least one of --pull-data or --add-h3-indices"

#     if pull_data:
#         run()
#     if add_h3_indices:
#         add_h3_indices()

@click.group()
def main():
    pass


main.add_command(add_h3_indices)

if __name__ == "__main__":

    main()