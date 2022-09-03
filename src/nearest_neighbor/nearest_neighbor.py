import gc
from pathlib import Path
from typing import Dict, List, Union

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

        # set the UTM epsg code
        self._utm_proj = pyproj.Proj("epsg:32619")
        # set the lat/lon epsg code
        self._latlon_proj = pyproj.Proj("epsg:4326")

        # store dataframe of hexagons that have been visited, to reduce read time
        self._visited_hexes = {}

        # store the utm polygons of the hexagons that have been visited, to reduce read time
        self._visited_utm_polygons = {}

    def _compose_table(self, hex_id: str) -> gpd.GeoDataFrame:
        """
        Compose a table of all shapes in the hexagon.
        """
        if hex_id in self._visited_hexes:
            return self._visited_hexes[hex_id]

        if not (self._data_dir / hex_id).exists():
            return None

        # just create a single dataframe for all tags
        df = pd.concat(
            [
                load_city_tag(
                    file.parent, file.stem, filter_values=self._filter
                ).to_crs(self._utm_proj.crs)
                for file in self._data_dir.joinpath(hex_id).iterdir()
                if (file.suffix == ".pkl") and (file.stem in self._tags)
            ]
        )

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

    def _get_utm_coords(self, lat: float, lon: float) -> List[float]:
        """
        Convert lat/lon coordinates to UTM coordinates.

        Args:
            lat: Latitude
            lon: Longitude
        """
        return self._utm_proj(
            lon,
            lat,
        )

    # def _get_latlon_coords(self, x: float, y: float) -> List[float]:
    #     """
    #     Convert UTM coordinates to lat/lon coordinates.

    #     Args:
    #         x: UTM x coordinate
    #         y: UTM y coordinate
    #     """
    #     return self._utm_proj(x, y, inverse=True)

    # #  below comes from https://autogis-site.readthedocs.io/en/latest/notebooks/L3/06_nearest-neighbor-faster.html
    def get_nearest(
        self,
        src_points,
        candidates,
        k_neighbors=10,
    ):
        # TODO: reinvestigate this function if needed
        """Find nearest neighbors for all source points from a set of candidate points"""

        # Create tree from the candidate points
        tree = KDTree(candidates, leaf_size=15, metric="minkowski")

        # Find closest points and distances
        indices = tree.query_radius(src_points, r=self._radius)

        # Transpose to get distances and indices into arrays
        # distances = distances.transpose()
        # indices = indices.transpose()

        # # Get closest indices and distances (i.e. array at index 0)
        # # note: for the second closest points, you would take index 1, etc.
        # closest = indices[0]
        # closest_dist = distances[0]

        # Return indices and distances
        return indices

    def nearest_neighbor(
        self, left_gdf, right_gdf, left_col=None, right_col=None, return_dist=False
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
        closest = self.get_nearest(src_points=left, candidates=right)

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
            if len(c):
                counts = pivoted.loc[c, :].sum(axis=0)
                records.append({'index': left_gdf.index[i], **counts.to_dict()})

        # return the records
        return pd.DataFrame(records).set_index('index')


    def _count_nearest_tags(
        self, hex_id: str, tag: str, indices: List[int]
    ) -> Dict[str, int]:
        """
        Count the nearest tags in a hexagon.

        Args:
            hex_id: Hexagon ID
            indices: Indices of nearest neighbors
        """
        # get the tags from the hexagon
        tags = self._visited_hexes[hex_id][tag][tag].iloc[indices].value_counts()
        return tags.to_dict()

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
        tag_columns = [
            f"{tag}_{s}" for tag, subset in self._filter.items() for s in subset
        ]
        gdf = pd.concat([gdf, pd.DataFrame(columns=tag_columns)], axis=1)
        # gdf.fillna(0, inplace=True)
        gdf["utm_centroid"] = gdf.geometry.to_crs(self._utm_proj.crs)
        gdf["circle_geom"] = gdf["utm_centroid"].buffer(self._radius)
        gdf.reset_index(drop=True, inplace=True)

        # Add the h3 5 level hexagon ID
        gdf["h3_5"] = gdf.h3.apply(h3.h3_to_parent, res=5)
        return gdf

    def count_tags(
        self, gdf: gpd.GeoDataFrame, hex_file_resolution: int = 5
    ) -> gpd.GeoDataFrame:  # sourcery skip: low-code-quality
        """
        This is applied on a groupby object, where the objects were grouped by their hexid

        Args:
            hex_id (str): _description_
            gdf (gpd.GeoDataFrame): _description_

        Returns:
            gpd.GeoDataFrame: _description_
        """
        #  create a copy for return
        gdf = gdf.copy()
        # hex_id = gdf.h3_5.iloc[0]
        # print(hex_id)
        hex_id = gdf.h3_5.iloc[0]
        print(hex_id)
        needed_hex = [
            hex_id,
        ]

        # check if all the circles are within the parent hexagon
        contained = gdf["circle_geom"].within(self._h3_to_utm_polygon(needed_hex[0]))
        non_contained = sum(~contained)
        if non_contained > 0:
            neighbors = [
                (h3_id, self._h3_to_utm_polygon(needed_hex[0]))
                for h3_id in h3.k_ring(hex_id, 1)
            ]
            for neighbor in neighbors:
                # you potentially don't need to check all the polygons
                if gdf["circle_geom"].overlaps(neighbor[1]).any():
                    needed_hex.append(neighbor[0])

        # for hex_id in needed_hex:
        tag_df = pd.concat([self._compose_table(hex_id) for hex_id in needed_hex], axis=0)
            # if tag not in value_counts[circle[0]].keys():
        #     value_counts[circle[0]] = {}
        # print("calculating for hex", hex_id)
        d = self.nearest_neighbor(
            left_gdf=gdf,
            right_gdf=tag_df,
            left_col="utm_centroid",
            right_col="utm_centroid",
        )

        # update the gdf with the values in d
        gdf.update(d)
 
        self._pop_non_neighbors(
            hex_id,
        )

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
