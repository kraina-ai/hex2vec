import os
import asyncio
from functools import partial, wraps
from pathlib import Path
from typing import Dict, Generator, List, Set, Union

import backoff
from tqdm import tqdm
import alphashape
import geopandas as gpd
import h3
import osmnx as ox
import pandas as pd
from shapely.geometry import mapping, MultiPolygon


from ..data.download import ensure_geometry_type
from ..data.load_data import load_city_tag, load_city_tag_h3
from ..data.make_dataset import h3_to_polygon
from ..settings import DATA_RAW_DIR
from src.utils.advanced_tags import Tag

# chunk list
def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))


# little of a hack to get around the fact that osmnx doesn't have async
def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return run


# cleaner way to get series of points into a list
def geometry_series_to_xy(geometry_series, epgs=32633):
    g = geometry_series.to_crs(epsg=epgs).copy()
    return list(zip(g.x, g.y))


def iterate_hex_dir(parent_dir: Path) -> Generator[Path, None, None]:
    for hex_id in parent_dir.iterdir():
        if h3.h3_is_valid(hex_id.stem) and hex_id.is_dir():
            yield hex_id


def cover_point_array_w_hex(
    point_array: pd.Series,
    resolution: int,
    epgs: int = 32633,
) -> Set[str]:

    xy = geometry_series_to_xy(point_array, epgs=epgs)
    print("computing alpha shape, this may take a while...")
    res = alphashape.alphashape(xy, alpha=0.001)
    res = res.buffer(2 * h3.edge_length(resolution=resolution, unit="m"))
    convex_hull_df = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(res),
        crs=f"EPSG:{epgs}",
    )
    convex_hull_df = convex_hull_df.to_crs(epsg=4326)
    feature = mapping(convex_hull_df)

    # reverse coordinates in geojson
    for feature in feature["features"]:
        geom = feature["geometry"]
        geom["coordinates"] = [[j[::-1] for j in i] for i in geom["coordinates"]]
        hexes = list(h3.polyfill(geom, resolution))
        break

    # convert the hex ids to polygons
    return gpd.GeoDataFrame(
        data={
            "h3": hexes,
        },
        geometry=gpd.GeoSeries(map(h3_to_polygon, hexes)),
        crs="EPSG:4326",
    )


@backoff.on_exception(backoff.expo, Exception, max_tries=8, max_time=300)
def ox_geometries(
    *args,
    **kwargs,
) -> gpd.GeoDataFrame:
    return ox.geometries_from_polygon(*args, **kwargs)


async_ox_geometries = async_wrap(ox_geometries)


def pull_hex_tags_synch(
    row: pd.Series,
    city_dir: Path,
    tag_list: str,
    simplify_data: bool = True,
    force_pull: bool = False,
) -> pd.Series:

    # make the directory
    hex_dir = city_dir.joinpath(row["h3"])
    hex_dir.mkdir(parents=True, exist_ok=True)
    # print("running for hex", row["h3"])
    for tag in tag_list:
        if not hex_dir.joinpath(f"{tag}.pkl").exists() or force_pull:
            gdf = ox_geometries(row["geometry"], tags={tag: True})
            # clean the data
            if not gdf.empty:
                gdf = ensure_geometry_type(gdf)
                gdf = (
                    gdf.reset_index()[["osmid", tag, "geometry"]]
                    if simplify_data
                    else gdf.reset_index()
                )
                # save the gdf
                gdf.to_pickle(
                    hex_dir.joinpath(f"{tag}.pkl").absolute(),
                )
        else:
            print(f"{tag} already exists")


async def walk_n_queue(
    queue: asyncio.Queue,
    city_dir: Path,
    hex_gdf: gpd.GeoDataFrame,
    tag_list: str,
    resolution: int,
    simplify_data: bool = True,
    force_pull: bool = False,
) -> None:

    r_dir = city_dir.joinpath(f"resolution_{resolution}")
    r_dir.mkdir(parents=True, exist_ok=True)
    for _, row in hex_gdf.iterrows():
        hex_dir = r_dir.joinpath(row["h3"])
        hex_dir.mkdir(parents=True, exist_ok=True)
        for tag in tag_list:
            if (
                not (
                    hex_dir.joinpath(f"{tag}.pkl").exists()
                    or hex_dir.joinpath(f"{tag}_is_empty.txt").exists()
                )
                or force_pull
            ):
                await queue.put((hex_dir.joinpath(f"{tag}.pkl"), row.geometry, tag))
                await asyncio.sleep(0)


# async def pull_tags_for_hex(
#     row: pd.Series,
#     city_dir: Path,
#     tag_list: str,
#     simplify_data: bool = True,
#     force_pull: bool = False
# ) -> pd.Series:


async def pull_tags_for_hex(queue: asyncio.Queue, semaphore: asyncio.Semaphore):
    simplify_data = True
    async with semaphore:
        while True:
            # this is probs bad practice
            save_path, geom, tag = await queue.get()
            print(
                f"pulling {save_path.parent.parent.parent.stem}/{save_path.parent.stem}/{tag}"
            )
            gdf = await async_ox_geometries(geom, tags={tag: True})
            # clean the data
            if not gdf.empty:
                gdf = ensure_geometry_type(gdf)
                gdf = (
                    gdf.reset_index()[["osmid", tag, "geometry"]]
                    if simplify_data
                    else gdf.reset_index()
                )
                # save the gdf
                gdf.to_pickle(save_path.absolute())
                print(
                    f"finished {save_path.parent.parent.parent.stem}/{save_path.parent.stem}/{tag}"
                )
            else:
                # record that the df is empty so we don't try again
                with open(
                    save_path.parent.joinpath(f"{tag}_is_empty.txt").absolute(), "w"
                ) as f:
                    pass
            # tell everyon the that the task is done
            queue.task_done()


async def pull_tags_for_hex_gdf(
    city_dir: Path,
    hex_gdf: gpd.GeoDataFrame,
    tag_list: str,
    resolution: int,
    simplify_data: bool = True,
    force_pull: bool = False,
) -> None:

    # make the directory
    r_dir = city_dir.joinpath(f"resolution_{resolution}")
    r_dir.mkdir(parents=True, exist_ok=True)
    for row in hex_gdf.iterrows():
        await pull_tags_for_hex(row[1], r_dir, tag_list, simplify_data, force_pull)
        # sleep to defer to other processes
        await asyncio.sleep(0.01)


def join_hex_dfs(
    hex_parent_dir: Path,
    tag_list: List[Tag],
    target_resolution: int,
    output_dir: Path,
    force_overwrite: bool = False,
) -> gpd.GeoDataFrame:

    # create a map of smaller hexes to larger hexes
    for hex_id in tqdm(list(iterate_hex_dir(hex_parent_dir))):
        # create a hexagon gpd
        target_hex_ids = list(h3.h3_to_children(hex_id.stem, target_resolution))
        all_hex_polygons = list(map(h3_to_polygon, target_hex_ids))
        hexes_gdf = gpd.GeoDataFrame(
            pd.DataFrame({"h3": target_hex_ids, "geometry": all_hex_polygons}),
            crs="EPSG:4326",
        )

        # create a list of other parent-level hexagons needed.
        # This is because the children are not always geographically contained.
        needed_hexes = [
            hex_id,
            *[hex_parent_dir / h for h in h3.k_ring(hex_id.stem, 1)],
        ]

        # create a list of the buffered boundary hexagons. It's not necessary to have the neighbors of these hexagons.
        boundary = False
        if (hex_parent_dir.parent / "boundary.hex.txt").exists():
            with open((hex_parent_dir.parent / "boundary.hex.txt"), "r") as f:
                boundary = hex_id.stem in f.read().split("\n")

        # check that all hexagons exist (they won't, because at some point we are at the edge of a hexagon)
        drops = []
        for i, p_hex in enumerate(needed_hexes):
            if not p_hex.exists() and not boundary:
                print(
                    f"{p_hex.stem} is needed to completely cover {hex_id.stem} children but missing"
                )
                drops.append(i)

        # drop the missing parent hexagons
        for i in drops[::-1]:
            needed_hexes.pop(i)

        # # load the tag data in parallel. This could get RAM heavy. Doesn't work on cluster
        ncpus = os.cpu_count()
        # Parallel(n_jobs=ncpus)(delayed(_map_2_small_hex)(target_resolution, output_dir, hex_id, hexes_gdf.copy(), needed_hexes, chunk_list) for chunk_list in chunker_list(tag_list, ncpus))
        # for tag in tag_list:
        _map_2_small_hex(
            target_resolution,
            output_dir,
            hex_id,
            hexes_gdf.copy(),
            needed_hexes,
            tag_list,
            force_overwrite,
        )


def _map_2_small_hex(
    target_resolution,
    output_dir,
    hex_id,
    hexes_gdf,
    needed_hexes,
    tag_list: List[Tag],
    force_overwrite: bool = False,
):
    save_location = output_dir.joinpath(
        hex_id.stem,
    )
    save_location.mkdir(parents=True, exist_ok=True)

    for tag in tag_list:
        # create the save path
        tag_save_path = save_location.joinpath(f"{tag.osmxtag}_{target_resolution}.pkl")

        # check if the file exists
        if tag_save_path.exists() and not force_overwrite:
            continue

        tag_dfs = []
        for p_hex in needed_hexes:
            # TODO: For synthetic tags, this should use the OSMNX tag and
            tag_gdf = load_city_tag(p_hex, tag=tag, data_dir=hex_id)
            if tag_gdf is not None:
                tag_dfs.append(tag_gdf)

        if len(tag_dfs):
            # create a hex + neighbors super df
            tag_gdf = pd.concat(tag_dfs, axis=0)
            # drop duplicated osmids
            tag_gdf = tag_gdf.drop_duplicates(subset=["osmid"])
            # spatial join of the hexes with the tag data. Only save data that is in the target hexagons.
            tag_gdf = gpd.sjoin(
                tag_gdf, hexes_gdf, how="inner", predicate="intersects"
            )[["h3", "osmid", tag.osmxtag, "geometry"]]
            # create a save location for the data
            tag_gdf.to_pickle(
                tag_save_path.absolute(),
                protocol=4,
            )


def group_h3_tags(
    hex_id: str,
    resolution: int,
    tags: List[Tag],
    fill_missing=True,
    data_dir: Path = DATA_RAW_DIR,
) -> pd.DataFrame:
    dfs = []
    unique_h3 = list(h3.h3_to_children(hex_id, resolution))

    for tag in tags:
        # create df
        df = load_city_tag_h3(hex_id, tag, resolution, data_path=data_dir)
        if df is not None and not df.empty:
            tag_grouped = tag.group_df_by_tag_values(df)
        else:
            tag_grouped = pd.DataFrame(data={"h3": unique_h3})

        # fill missing values, using pandas concat
        if fill_missing and tag.filter_values is not None:
            #
            columns_names = [
                col for col in tag.columns if col not in tag_grouped.columns
            ]
            tag_grouped = pd.concat(
                [tag_grouped, pd.DataFrame(columns=columns_names)],
                axis=1,
                verify_integrity=True,
            )

        dfs.append(tag_grouped)

    results = pd.concat(
        dfs,
        axis=0,
    )

    # fill missing values and sum the results for each h3
    agg_dict = {}
    for tag in tags:
        agg_dict.update(tag.agg_dict(results, level="h3"))

    results = results.fillna(0).groupby("h3").agg(agg_dict)
    return results.reindex(index=unique_h3, fill_value=0).reset_index()


def group_hex_tags(
    hex_parent_dir: Path,
    tags: List[Tag],
    output_dir: Path,
    resolution: int,
) -> None:

    for hex_id in tqdm(list(iterate_hex_dir(hex_parent_dir))):
        h3_grouped_df = group_h3_tags(
            hex_id=hex_id.stem,
            resolution=resolution,
            tags=tags,
            data_dir=hex_id.parent,
        )
        save_location = output_dir.joinpath(
            hex_id.stem,
        )
        save_location.mkdir(parents=True, exist_ok=True)
        h3_grouped_df.to_pickle(
            save_location.joinpath(f"{resolution}.pkl").absolute(),
            protocol=4,
        )


def create_city_from_hex(
    hex_parent_dir: Path, output_dir: Path, resolution: int, drop_all_zero=True
) -> None:
    dfs = [
        pd.read_pickle(hex_id.joinpath(f"{resolution}.pkl"))
        for hex_id in iterate_hex_dir(hex_parent_dir)
    ]

    df = pd.concat(dfs, ignore_index=True).set_index("h3")
    df.fillna(0, inplace=True)
    if drop_all_zero:
        df = df[(df.T != 0).any()]
    df.to_pickle(output_dir.joinpath(f"{resolution}.pkl").absolute(), protocol=4)
    return df


def get_buffer_hexes(
    hexes: Set[str], save_boundary: bool = True, save_path: Path = None
) -> Set[str]:
    reported_hex = set()
    for hex in hexes:
        # find the missing and add them
        if missing := h3.k_ring(hex, 1) - hexes:
            for _h in missing:
                reported_hex.add(_h)

    if save_boundary:
        if (save_path / "boundary.hex.txt").exists():
            with open(save_path / "boundary.hex.txt", "r") as f:
                already_marked = {h for h in f.read().split("\n") if len(h)}
        else:
            already_marked = set()

        with open(save_path / "boundary.hex.txt", "w") as f:
            f.write("\n".join(reported_hex.union(already_marked)))

    return reported_hex


def fetch_city_h3s(
    city_name: Union[str, List[str]],
    return_gdf: bool = False,
    convex_hull: bool = False,
    res: int = None,
) -> Dict:
    # sourcery skip: raise-specific-error
    from osmnx.geocoder import _geocode_query_to_gdf
    from shapely.geometry import Polygon

    if not return_gdf:
        assert (
            res is not None
        ), "You must either call for the return of the gdf or specify a resolution"

    try:
        # bypass osmnx extras
        if isinstance(city_name, str):
            city_gdf = _geocode_query_to_gdf(
                city_name, by_osmid=False, which_result=None
            )
        else:
            city_gdf = pd.concat(
                (
                    _geocode_query_to_gdf(c, by_osmid=False, which_result=None)
                    for c in city_name
                )
            )

        if return_gdf:
            return city_gdf

        # if use the convex hull of geometry if specified
        if convex_hull:  # or isinstance(city_gdf.geometry.iloc[0], MultiPolygon):
            city_gdf.geometry = city_gdf.geometry.convex_hull

        # if the city has a multi-polygon geometry, then explod into individual polygons
        if any(city_gdf.geometry.map(lambda x: isinstance(x, MultiPolygon))):
            city_gdf = city_gdf.explode(index_parts=False).reset_index()

        h3s = set()

        def h3_mapper(poly: Polygon) -> None:
            poly = mapping(poly)
            # have to reverse the coordinates...
            poly["coordinates"] = [
                [c[::-1] for c in coords] for coords in poly["coordinates"]
            ]
            _h3s = h3.polyfill(poly, res=res)
            h3s.update(_h3s)

        city_gdf.geometry.apply(h3_mapper)
        return h3s

        # # reverse the coordinates for the h3 api
        # city_boundary_geojson['coordinates'] = [[c[::-1] for c in coords] for coords in city_boundary_geojson['coordinates']]
        # # return the json
        # return city_boundary_geojson

    except (IndexError, KeyError) as e:

        raise Exception(
            f"OSMNX did not find a boundary geometry for {city_name}"
        ) from e
