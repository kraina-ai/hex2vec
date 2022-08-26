from src.data.download import download_whole_city
from src.settings import DATA_RAW_DIR
from tqdm import tqdm



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


if __name__ == "__main__":

    run()