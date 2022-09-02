import gc
from pathlib import Path
from typing import Dict, List, Union

import h3
import pyproj
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
from shapely.geometry import Point, Polygon
from src.data.load_data import load_city_tag, load_filter

from src.data.make_dataset import h3_to_polygon
from src.utils.tesselate_amazon_data import group_h3_tags


class NearestNeighbor:
    def __init__(self, data_dir: Path, filter_path: Path, radius: float) -> None:
        """

        Args:
            data_dir (Path): Path to the data directory
            filter_path (Path): path to the filter file
            radius (float): should be in meters
        """
        self._data_dir = data_dir
        self._filter = load_filter(filter_path)
        self._radius = radius

        # set the UTM epsg code
        self._utm_proj = pyproj.Proj("epsg:32619")
        # set the lat/lon epsg code
        self._latlon_proj = pyproj.Proj("epsg:4326")

        # store dataframe of hexagons that have been visited, to reduce read time
        self._visited_hexes = {}

        # store the utm polygons of the hexagons that have been visited, to reduce read time
        self._visited_utm_polygons = {}

    def _compose_table(self, hex_id: str) -> Dict[str, gpd.GeoDataFrame]:
        """
        Compose a table of all shapes in the hexagon.
        """
        if hex_id in self._visited_hexes:
            return self._visited_hexes[hex_id]

        if not (self._data_dir / hex_id).exists():
            return {}

        dfs = {
            file.stem: load_city_tag(file.parent, file.stem).to_crs(self._utm_proj.crs)
            for file in self._data_dir.joinpath(hex_id).iterdir()
            if file.suffix == ".pkl"
        }
        self._visited_hexes[hex_id] = dfs
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
    # def get_nearest(self, src_points, candidates, k_neighbors=20, radius=30):
    #     #TODO: reinvestigate this function if needed
    #     """Find nearest neighbors for all source points from a set of candidate points"""

    #     # Create tree from the candidate points
    #     tree = BallTree(candidates, leaf_size=15, metric='haversine')

    #     # Find closest points and distances
    #     distances, indices = tree.query(src_points, k=k_neighbors)

    #     # Transpose to get distances and indices into arrays
    #     distances = distances.transpose()
    #     indices = indices.transpose()

    #     # Get closest indices and distances (i.e. array at index 0)
    #     # note: for the second closest points, you would take index 1, etc.
    #     closest = indices[0]
    #     closest_dist = distances[0]

    #     # Return indices and distances
    #     return (closest, closest_dist)

    # def _count_nearest_tags(self, hex_id: str, tag: str, indices: List[int]) -> Dict[str, int]:
    #     """
    #     Count the nearest tags in a hexagon.

    #     Args:
    #         hex_id: Hexagon ID
    #         indices: Indices of nearest neighbors
    #     """
    #     # get the tags from the hexagon
    #     tags = self._visited_hexes[hex_id][tag][tag].iloc[indices].value_counts()
    #     return tags.to_dict()

    # def nearest_neighbor(self, left_gdf, right_gdf, left_col=None, right_col=None, return_dist=False):
    #     """
    #     For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    #     NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    #     """
    #     left_geom_col = left_col or left_gdf.geometry.name
    #     right_geom_col = right_col or right_gdf.geometry.name

    #     # Ensure that index in right gdf is formed of sequential numbers
    #     right = right_gdf.copy().reset_index(drop=True)

    #     # Parse coordinates from points and insert them into a numpy array as RADIANS
    #     # Notice: should be in Lat/Lon format
    #     left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    #     right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())

    #     # Find the nearest points
    #     # -----------------------
    #     # closest ==> index in right_gdf that corresponds to the closest point
    #     # dist ==> distance between the nearest neighbors (in radians)
    #     closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    #     # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    #     closest_points = right.loc[closest]

    #     # Ensure that the index corresponds the one in left_gdf
    #     closest_points = closest_points.reset_index(drop=True)

    #     # Add distance if requested
    #     if return_dist:
    #         # Convert to meters from radians
    #         earth_radius = 6371000  # meters
    #         closest_points['distance'] = dist * earth_radius

    #     return closest_points

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
        gdf["circle_geom"] = gdf.geometry.to_crs(self._utm_proj.crs).buffer(
            self._radius
        )
        gdf.reset_index(drop=True, inplace=True)
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
        hex_id = gdf.h3.iloc[0]
        print(hex_id)
        needed_hex = [
            h3.h3_to_parent(hex_id, hex_file_resolution),
        ]
        # check if all the circles are within the parent hexagon
        # if not, read in the others that we need
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

        value_counts = {}
        # for each tag, count the number of points in each hexagon. will have to think of a more efficient way to do this
        for circle in gdf[["circle_geom"]].itertuples():
            value_counts[circle[0]] = {}
            for hex_id in needed_hex:
                for tag, tag_df in self._compose_table(hex_id).items():
                    # if tag not in value_counts[circle[0]].keys():
                    #     value_counts[circle[0]] = {}

                    values = tag_df.loc[
                        tag_df["geometry"].intersects(circle[1])
                        | tag_df["geometry"].within(circle[1]),
                        ["osmid", tag],
                    ].to_dict("records")

                    if len(values) > 0:
                        for _d in values:
                            osmid, value = _d["osmid"], _d[tag]
                            if value in self._filter[tag]:
                                tag_name = f"{tag}_{value}"
                                if tag_name not in value_counts[circle[0]].keys():
                                    value_counts[circle[0]][tag_name] = [osmid]
                                else:
                                    value_counts[circle[0]][tag_name].append(osmid)

        # sum the counts for each tag
        value_counts = {
            k: {k2: len(set(v2)) for k2, v2 in v.items()}
            for k, v in value_counts.items()
        }

        # add the counts to the dataframe
        new_df = pd.DataFrame(value_counts).T

        # merge the new dataframe with the old one
        gdf[new_df.columns] = new_df

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
