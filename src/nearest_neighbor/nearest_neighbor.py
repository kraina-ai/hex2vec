import gc
from pathlib import Path
from typing import Dict, List, Union
import asyncio

import h3
import pyproj
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree, KDTree
from shapely.geometry import Point, Polygon
from src.data.load_data import load_city_tag, load_filter

from src.data.make_dataset import h3_to_polygon
from src.utils.tesselate_amazon_data import group_h3_tags
from src.utils.tesselate_amazon_data import pull_hex_tags_synch, pull_tags_for_hex


class NearestNeighbor:
    def __init__(
        self, data_dir: Path, filter_path: Path, radius: float, tags: List[str]
    ) -> None:
        """

        Args:
            data_dir (Path): Path to the data directory
            filter_path (Path): path to the filter file
            radius (float): should be in meters
        """
        self._data_dir = data_dir
        self._filter = load_filter(filter_path)
        self._radius = radius
        self._tags = tags
        self._h3_level = 5

        self._tag_columns = [
            f"{tag}_{s}" for tag, subset in self._filter.items() for s in subset
        ]

        # find all hexes
        self._all_hexes = [
            dir.stem
            for dir in self._data_dir.iterdir()
            if h3.h3_is_valid(dir.stem)
            and h3.h3_get_resolution(dir.stem) == self._h3_level
        ]

        # set the UTM epsg code
        self._utm_proj = pyproj.Proj("epsg:32619")
        # set the lat/lon epsg code
        self._latlon_proj = pyproj.Proj("epsg:4326")

        # store dataframe of hexagons that have been visited, to reduce read time
        self._visited_hexes = {}

        # store the utm polygons of the hexagons that have been visited, to reduce read time
        self._visited_utm_polygons = {}

        # store the tree in cache
        self._hex_tree = {}

    def _compose_tree(self, hex_id: str, candidates: np.ndarray) -> KDTree:
        if hex_id in self._hex_tree:
            return self._hex_tree[hex_id]

        self._hex_tree[hex_id] = KDTree(candidates, leaf_size=40, metric="minkowski")
        return self._hex_tree[hex_id]

    def _compose_table(self, hex_id: str) -> gpd.GeoDataFrame:
        """
        Compose a table of all shapes in the hexagon.
        """
        if hex_id in self._visited_hexes:
            return self._visited_hexes[hex_id]

        if not (self._data_dir / hex_id).exists():
            return None

        # just create a single dataframe for all tags
        dfs = [
                load_city_tag(
                    file.parent, file.stem, filter_values=self._filter
                ).to_crs(self._utm_proj.crs)
                for file in self._data_dir.joinpath(hex_id).iterdir()
                if (file.suffix == ".pkl") and (file.stem in self._tags)
            ]

        if len(dfs):
            df = pd.concat(
                dfs
            )
        else:
            return None

        # df.fillna("", inplace=True)

        # create a centroid column for each of the points
        df["utm_centroid"] = df.geometry.centroid

        # # sort by the distance to the center of the hexagon
        # df["distance"] = df.center.distance(
        #     Point(self._get_utm_coords(*h3.h3_to_geo(hex_id)))
        # )
        # df.sort_values(by="distance",).reset_index(
        #     drop=True
        # ).drop("distance", axis=1, inplace=True)

        self._visited_hexes[hex_id] = df
        return self._visited_hexes[hex_id]

    def get_traversal_order(
        self, point_array: gpd.GeoSeries, unique_hexes: gpd.GeoSeries
    ) -> List[str]:
        # start at center and then ring around
        tmp = point_array.to_crs(self._utm_proj.crs).copy(deep=True)
        center = tmp.y.mean(), tmp.x.mean()
        center_hex = h3.geo_to_h3(
            *self._get_utm_coords(*center, inverse=True)[::-1], self._h3_level
        )
        t_list = [center_hex]
        to_visit = [hex for hex in self._all_hexes if hex in unique_hexes]
        i = 0
        while len(tmp) and len(to_visit):
            hex_poly = self._h3_to_utm_polygon(t_list[-1])
            contained = tmp.within(hex_poly)
            tmp = tmp[~contained]

            to_visit.pop(to_visit.index(t_list[-1]))
            found = False

            # try to visit "most central hex" with remaining neighbors.
            for h in t_list:
                found = False
                for n in to_visit:
                    if h3.h3_indexes_are_neighbors(h, n) and n not in t_list:
                        break
                if found:
                    break

            for n in h3.k_ring(h):
                if n in to_visit:
                    t_list.append(n)
                    found = True
                    break
            if not len(to_visit):
                break
            if not found:
                t_list.append(to_visit[-1])
        return t_list

    def _pop_non_neighbors(self, center_hex: str) -> List[int]:
        """
        Remove hexagons that are not neighbors of the active hexagon.

        Args:
            hexes: List of hexagons
            center_hex: Center hexagon
        """
        neighbors = h3.k_ring(center_hex, 1)
        self._visited_hexes = {
            hex_id: self._visited_hexes[hex_id]
            for hex_id in self._visited_hexes.keys()
            if hex_id in neighbors
        }
        gc.collect()

    @property
    def utm_proj(self, num: Union[int, str]) -> None:
        """
        Set the UTM epsg code.
        """
        self._utm_proj = pyproj.Proj(init=f"epsg:{num}")

    @property
    def latlon_proj(self, num: Union[int, str]) -> None:
        """
        Set the lat/lon epsg code.
        """
        self._latlon_proj = pyproj.Proj(init=f"epsg:{num}")

    def _get_utm_coords(self, lat: float, lon: float, inverse=False) -> List[float]:
        """
        Convert lat/lon coordinates to UTM coordinates.

        Args:
            lat: Latitude
            lon: Longitude
        """
        return self._utm_proj(lon, lat, inverse=inverse)

    # #  below comes from https://autogis-site.readthedocs.io/en/latest/notebooks/L3/06_nearest-neighbor-faster.html
    def get_nearest(self, src_points, candidates, hex_id):

        # Create tree from the candidate points
        tree = self._compose_tree(hex_id, candidates)

        # Return indices and distances
        return tree.query_radius(src_points, r=self._radius)

    def nearest_neighbor(
        self,
        left_gdf,
        right_gdf,
        hex_id,
        left_col=None,
        right_col=None,
        return_dist=False,
    ):
        """
        For each point in left_gdf, find closest point in right GeoDataFrame and return them.

        NOTICE: Assumes that the input Points are in UTM projection, and are in the same projection.
        """
        left_geom_col = left_col or left_gdf.geometry.name
        right_geom_col = right_col or right_gdf.geometry.name
        # Ensure that index in right gdf is formed of sequential numbers
        right_gdf = right_gdf.reset_index(drop=True)
        # Parse coordinates from points and insert them into a numpy array as RADIANS
        # Notice: should be in Lat/Lon format
        left = np.array(
            list(zip(left_gdf[left_geom_col].x, left_gdf[left_geom_col].y))
            # .apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180))
            # .to_list()
        )
        right = np.array(
            list(zip(right_gdf[right_geom_col].x, right_gdf[right_geom_col].y))
            # .apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180))
            # .to_list()
        )

        # Find the nearest points
        # -----------------------
        # closest ==> index in right_gdf that corresponds to the closest point
        # dist ==> distance between the nearest neighbors (in radians)
        closest = self.get_nearest(src_points=left, candidates=right, hex_id=hex_id)

        # iterate over the closest points and create a new dataframe
        columns = right_gdf.columns
        all_indices = []
        for c in closest:
            all_indices.extend(c)

        pivoted = pd.get_dummies(
            right_gdf.loc[all_indices, [tag for tag in self._tags if tag in columns]]
        )

        # sum for each of the groups:
        records = []
        for i, c in enumerate(closest):
            # if len(c):
            counts = pivoted.loc[c, :].sum(axis=0)
            records.append({"index": left_gdf.index[i], **counts.to_dict()})

        # return the records
        return pd.DataFrame(records).set_index("index")

    # def _get_nearest_tags(self, hex_id: str, tag: str, indices: List[int]) -> Dict[str, int]:
    def prepare_df_apply(
        self, gdf: gpd.GeoDataFrame, inline: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Prepare the GeoDataFrame for applying the nearest neighbor function.

        Args:
            gdf: GeoDataFrame
            tag: Tag to be used
        """
        # get the tags from the hexagon
        gdf = pd.concat([gdf, pd.DataFrame(columns=self._tag_columns)], axis=1)
        # gdf.fillna(0, inplace=True)
        gdf["utm_centroid"] = gdf.geometry.to_crs(self._utm_proj.crs)
        gdf["circle_geom"] = gdf["utm_centroid"].buffer(self._radius)
        gdf.reset_index(drop=True, inplace=True)

        # Add the h3 5 level hexagon ID
        gdf[f"h3_{self._h3_level}"] = gdf.h3.apply(h3.h3_to_parent, res=self._h3_level)
        return gdf

    def _pull_missing_hex(self, hex_id) -> None:
        # try:
        #     asyncio.get_event_loop().run_until_complete(
        #         pull_tags_for_hex(
        #             row={"h3": hex_id},
        #             city_dir=self._data_dir,
        #             tag_list=self._tags,
        #         )
        #     )
        # except RuntimeError as e:
        pull_hex_tags_synch(
            row={"h3": hex_id, "geometry": h3_to_polygon(hex_id)},
            city_dir=self._data_dir,
            tag_list=self._tags,
        )
        
        self._all_hexes.append(hex_id)

    def count_tags(
        self, gdf: gpd.GeoDataFrame, hex_file_resolution: int = 5
    ) -> gpd.GeoDataFrame:
        """
        This is applied on a groupby object, where the objects were grouped by their hexid

        Args:
            hex_id (str): _description_
            gdf (gpd.GeoDataFrame): _description_

        Returns:
            gpd.GeoDataFrame: _description_
        """
        gdf = gdf.copy()
        hex_id = gdf[f"h3_{self._h3_level}"].iloc[0]
        print(hex_id)
        needed_hex = [(hex_id, [True] * len(gdf))]
        contained = gdf["circle_geom"].within(self._h3_to_utm_polygon(needed_hex[0][0]))

        non_contained = sum(~contained)
        if non_contained > 0:
            neighbors = [
                (h3_id, self._h3_to_utm_polygon(needed_hex[0][0]))
                for h3_id in h3.k_ring(hex_id, 1)
            ]

            for neighbor in neighbors:
                check_idx = gdf["circle_geom"].overlaps(neighbor[1])
                if check_idx.any():
                    if neighbor[0] not in self._all_hexes:
                        print(
                            f"{neighbor[0]} needed but not in files. Pulling from API"
                        )
                        self._pull_missing_hex(neighbor[0])
                    needed_hex.append((neighbor[0], check_idx))
        ds = []
        tags = []
        for hex_id, check_idx in needed_hex:
            tag_df = self._compose_table(hex_id)
            if tag_df is not None:
                tags.extend(list(tag_df.columns))
                ds.append(
                    self.nearest_neighbor(
                        left_gdf=gdf.loc[check_idx],
                        right_gdf=tag_df,
                        left_col="utm_centroid",
                        right_col="utm_centroid",
                        hex_id=hex_id,
                    )
                )

        _tmp = pd.concat(ds, axis=0).fillna(0).reset_index()
        d = _tmp.groupby(["index"])[_tmp.columns.intersection(self._tag_columns)].sum()
        gdf.update(d)
        self._pop_non_neighbors(hex_id)
        return gdf

    def _h3_to_utm_polygon(self, hex: int) -> Polygon:
        if hex not in self._visited_utm_polygons:
            self._visited_utm_polygons[hex] = Polygon(
                [self._get_utm_coords(x, y) for [x, y] in h3.h3_to_geo_boundary(hex)]
            )
        return self._visited_utm_polygons[hex]


# def get_hexes(circle: Polygon, hexagon: Polygon, hex_id: str) -> List[str]:
#     """
#     Check if hexagon is big enough to contain circle centered on lat/lon. If not, return list of hex_ids that do.
#     """
#     neighbors = h3.k_ring(hex_id, 1)
#     return [neighbor for neighbor in neighbors if h3_to_polygon(neighbor).intersects(circle)]


# def latlon_to_utm(lat: float, lon: float) -> tuple:
#     """
#     Convert lat/lon to UTM coordinates.
#     """
#     return pyproj.transform(pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='epsg:32633'), lon, lat)


# def create_hex_df(hex_id: str, hex_id_df: ) -> gpd.GeoDataFrame:
#     """
#     Create a single geodataframe for all points.
#     """
#     hex_bbox = h3_to_polygon(hex_id)

#     all_hexes = hex_id_df[['geometry', 'h3']].apply()
