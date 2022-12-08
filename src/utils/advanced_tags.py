from typing import List
from typing import Dict
import pandas as pd
import geopandas as gpd
import re
import pyproj

# def _keep_dig(char):
#     """Keep only digits in a string."""
#     return char == "." or str.isdigit(char)

# def string_2_float(string):
#     """Filter digits in a string and convert to float."""
#     """ Used for Building Height """
#     return float("".join(filter(_keep_dig, string)))
# def pyproj_utm_crs(bbox: ) -> pyproj.CRS:
#     """Get the UTM CRS for a given latitude/longitude."""

#     utm_crs_list = pyproj.database.query_utm_crs_info(
#         datum_name="WGS 84",
#         area_of_interest=AreaOfInterest(
#             west_lon_degree=-93.581543,
#             south_lat_degree=42.032974,
#             east_lon_degree=-93.581543,
#             north_lat_degree=42.032974,
#         ),
#     )

    


def feet_and_inches_to_meters(x):
    # Use regular expressions to extract the feet and inches values from the string
    # extract feet and inches, where inches is optional
    feet, _, inch = re.findall(r"([\d\.]+)'(([\d\.]+)\")?", x)[0]
    feet = float(feet)
    inch = float(inch) if inch else 0
    x = feet + inch / 12
    # Convert feet and inches to meters
    return x * 0.3048


def feet_or_meters(width):
    # this is tricky, as we must infer whether the author inteded to use feet or meters
    # we will assume that if the value is less than 15, it is in meters
    # otherwise, we will assume it is in feet
    if width < 15:
        return width
    else:
        return width * 0.3048


def robust_null_checker(x: pd.Series) -> pd.Series:
    """Check if a value is null or not."""
    return x.isnull() | x.isna() | (x == "None") | (x == "none") | (x == "NONE")


class Tag:
    def __init__(self, tag: str, filter_values: List[str] = None) -> None:
        if "." in tag:
            self._tag = tag.split(".")[0]
            self._subtag = tag.split(".")[1]
            self._other_tags = tag.split(".")[2:]
        else:
            self._tag = tag
            self._subtag = None
            self._other_tags = (None,)

        # store filter values as set for faster lookup
        self.filter_values = set(filter_values) if filter_values else None

        # this is for the intermediate step. Basically do we need geometry after we have mapped the object to the hexagons
        self.geom_required = False

        # set this if you want the other tag columns to come with the raw data
        self._simplify_raw = True

    @property
    def keep_columns(self, ) -> List[str]:
        return ["osmid", self.osmxtag, "geometry"]

    def __str__(self) -> str:
        return self.tag

    def file_name(self, add_ons: str, ext: str) -> str:
        return f"{self.osmxtag}{add_ons}.{ext}"

    @property
    def dtype(self) -> str:
        return "int16"

    @property
    def osmxtag(
        self,
    ) -> str:
        return self._tag

    @property
    def tag(
        self,
    ) -> str:
        return (
            ".".join([self._tag, self._subtag, *self._other_tags])
            if self._subtag
            else self._tag
        )

    @property
    def columns(
        self,
    ) -> List[str]:
        if self.filter_values is not None:
            return [f"{self.tag}_{value}" for value in self.filter_values]
        else:
            return [self.tag]

    def extract_osmnx_tag(self, df_raw: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df_raw[["osmid", self.tag, "geometry"]] if self._simplify_raw else df_raw

    def group_df_by_tag_values(
        self,
        df,
    ):
        tmp = df.reset_index(drop=True)[["h3", self.tag]].copy(deep=True)
        indicators = (
            pd.get_dummies(tmp, columns=[self.tag], prefix=self.tag)
            .groupby("h3")
            .sum()
            .astype("int16")
        )
        return indicators.reset_index()

    def filter_df_by_tag_values(
        self,
        df,
    ) -> pd.DataFrame:
        if self.filter_values is not None:
            df = df[df[self.tag].isin(self.filter_values)]
        return df

    def agg_dict(self, df, level: str = "h3") -> Dict[str, str]:
        return {col: "sum" for col in df.columns.intersection(self.columns)}

    def sjoin(
        self,
        hexes_gdf,
        tag_gdf,
    ):
        """Spatial join between hexagons and tag_gdf"""
        return tag_gdf.sjoin(hexes_gdf, how="inner", predicate="intersects",).compute()[
            [
                "h3",
                "osmid",
                self.osmxtag,
                *(("geometry",) if self.geom_required else ()),
            ]
        ]


class BuildingArea(Tag):
    def __init__(self, tag: str = "building.area", *args, **kwargs) -> None:
        super().__init__(tag, *args, **kwargs)

        self.geom_required = True

    @property
    def dtype(self) -> str:
        return "float32"

    def file_name(self, add_ons: str, ext: str) -> str:
        return f"{self.tag}{add_ons}.{ext}"

    def _create_area_column(self, df):
        # create the area column
        # convert the crs to meters
        df[self.tag] = df["geometry"].to_crs(df["geometry"].estimate_utm_crs()).area
        return df

    def group_df_by_tag_values(self, df):
        # create the area column
        # convert the crs to meters
        df = self._create_area_column(df.copy(deep=True))
        # drop zero values
        df = df[df[self.tag] > 0]
        tmp = df.reset_index(drop=True)[["h3", self.tag]].copy(deep=True)
        indicators = tmp.groupby("h3").sum().astype("float32")
        return indicators.reset_index()

    def filter_df_by_tag_values(self, df):
        # nothing to do here
        return df

    def agg_dict(self, df, level: str = "h3"):
        # nothing custom to go here. sum is okay
        return super().agg_dict(df, level=level)

    def sjoin(
        self,
        hexes_gdf,
        tag_gdf,
    ):
        """
        Spatial join between hexagons and tag_gdf

        This overrides the default sjoin method because we need to find the intersection area,
        not just if the hexagon intersects the building
        """
        # filter out any buildings that are not polygons or multipolygons
        tag_gdf = tag_gdf.compute()
        tag_gdf = tag_gdf[
            tag_gdf["geometry"].geom_type.isin(["Polygon", "MultiPolygon"])
        ]

        return gpd.overlay(tag_gdf, hexes_gdf, how="intersection")[
            ["h3", "osmid", self.osmxtag, "geometry"]
        ]


class ParkingArea(Tag):
    def __init__(self, tag: str = "parking.area", *args, **kwargs) -> None:
        super().__init__(tag, *args, **kwargs)

        self._filter_values = {
            "bicycle_parking",
            "motorcycle_parking",
            "parking",
            "parking_entrance",
            "parking_space",
        }

        self.geom_required = True

    def file_name(self, add_ons: str, ext: str) -> str:
        return f"{self.tag}{add_ons}.{ext}"

    @property
    def dtype(self) -> str:
        return "float32"

    @property
    def osmxtag(self) -> str:
        return "amenity"
    
    def _create_area_column(self, df):
        # create the area column
        # convert the crs to meters
        try:
            df[self.tag] = df["geometry"].to_crs(df["geometry"].estimate_utm_crs()).area
        except:
            if len(df) < 1:
                df[self.tag] = 0
            else:
                raise ValueError("Could not convert to UTM")
        return df

    def group_df_by_tag_values(self, df):
        # create the area column
        # convert the crs to meters
        # filter out vlues that we don't want
        df = df.loc[df[self.osmxtag].isin(self._filter_values)].copy()
        df = self._create_area_column(df)
        # drop zero values
        df = df[df[self.tag] > 0]
        tmp = df.reset_index(drop=True)[["h3", self.tag]].copy(deep=True)
        indicators = tmp.groupby("h3").sum().astype("float32")
        return indicators.reset_index()

    def filter_df_by_tag_values(self, df):
        return df

    def agg_dict(self, df, level: str = "h3"):
        return super().agg_dict(df, level=level)

    def sjoin(
        self,
        hexes_gdf,
        tag_gdf,
    ):
        """
        Spatial join between hexagons and tag_gdf

        This overrides the default sjoin method because we need to find the intersection area,
        not just if the hexagon intersects the building
        """
        # filter out any buildings that are not polygons or multipolygons
        tag_gdf = tag_gdf.compute()
        tag_gdf = tag_gdf[
            tag_gdf["geometry"].geom_type.isin(["Polygon", "MultiPolygon"])
        ]

        return gpd.overlay(tag_gdf, hexes_gdf, how="intersection")[
            ["h3", "osmid", self.osmxtag, "geometry"]
        ]


class HighwayArea(BuildingArea):
    def __init__(self, tag: str = "highway.area", *args, **kwargs) -> None:
        super().__init__(tag, *args, **kwargs)
        self.geom_required = True
        self._filter_values = {
            'residential',
            'tertiary',
            'trunk',
            'primary',
            'secondary',
            'highway',
            'motorway',
            'service'
        }

        self._width_map = {
            'residential' : 10,
            'tertiary': 10,
            'trunk': 10,
            'primary': 10,
            'secondary': 10,
            'highway': 10,
            'motorway': 10,
            'service': 10
        }

        self._simplify_raw = False

    @property
    def osmxtag(self) -> str:
        return "highway"
    
    @staticmethod
    def _width_fixer(x: str, feet_prob: bool = False) -> float:
        try:
            # write regex to remove all non-numeric characters or characters that are not a period, "'", ''', or a space. terminate on a ;
            # see if ";" is in the string, is so, apply _width_fixer to all the values
            # replace all " "  with ""
            x = x.replace(' ', '')
            if ';' in x:
                return sum([HighwayArea._width_fixer(y, feet_prob=True) for y in x.split(';')])
            x = re.sub(r'[^0-9\.\'\"\sm]', '', x)
            if ('"' in x) or ("'" in x):
                return feet_and_inches_to_meters(x)
            if 'm' in x:
                return float(x.replace('m', ''))
            return feet_or_meters(float(x)) if not feet_prob else float(x) * 0.3048
        except:
            print(x)
            return 0

    @staticmethod
    def _lane_fixer(x: str) -> float:
        try:
            if ';' in x:
                return sum([HighwayArea._lane_fixer(y) for y in x.split(';')])
            return float(x)
        except:
            return 0        

    def _add_area(self, df):
        # get the utm zone
        zone = df.geometry.estimate_utm_crs()
        df['utm_geom'] = df.geometry.to_crs(zone)
        
        # add a width column
        # print(df['width'].value_counts())
        df['meter_width'] = 0

        null_width = robust_null_checker(df['width'])
        df.loc[~null_width, 'meter_width'] = df.loc[~null_width, 'width'].apply(self._width_fixer).astype('float')
        # where the width is not set, try the lane count * the average width of a lane
        null_lanes = robust_null_checker(df['lanes'])
        df.loc[(~null_lanes & null_width), 'meter_width'] = df.loc[(~null_lanes & null_width), 'lanes'].apply(self._lane_fixer).astype('float') * 3.5
        # where the width is still not set, try the highway type
        width_map = df[df['meter_width'] > 0].groupby('highway')['meter_width'].mean().to_dict()
        for k, v in self._width_map.items():
            if k not in width_map:
                width_map[k] = v
        df.loc[df['meter_width'] == 0, 'meter_width'] = df.loc[df['meter_width'] == 0, 'highway'].map(width_map)

        # buffer the geometry by the width, with a flat cap
        df['geometry'] = df['utm_geom'].buffer(df['meter_width'] / 2, cap_style=2).to_crs(df.geometry.crs)

        return df[['osmid', 'highway', 'geometry']]

    def sjoin(
        self,
        hexes_gdf,
        tag_gdf,
    ):
        """
        Spatial join between hexagons and tag_gdf

        This overrides the default sjoin method because we need to find the intersection area,
        not just if the hexagon intersects the building
        """
        # filter out any buildings that are not polygons or multipolygons
        tag_gdf = tag_gdf.compute()

        # filter out the values we don't want
        tag_gdf = tag_gdf.loc[tag_gdf[self.osmxtag].isin(self._filter_values)].copy()

        # tag_gdf is not empty
        if len(tag_gdf) > 0:
            tag_gdf = self._add_area(tag_gdf)
            tag_gdf = tag_gdf[
                tag_gdf["geometry"].geom_type.isin(["Polygon", "MultiPolygon"])
            ]


        return gpd.overlay(tag_gdf, hexes_gdf, how="intersection")
    
    def extract_osmnx_tag(self, df_raw: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        df_raw = df_raw[["osmid", self.osmxtag, "geometry", *df_raw.columns.intersection(['width', 'lanes'])]]
        return df_raw

    @property
    def keep_columns(self, ) -> List[str]:
        return super().keep_columns + ['width', 'lanes']



def build_tag(tag: str, *args, **kwargs) -> Tag:
    try:
        return {"building.area": BuildingArea, "parking.area": ParkingArea, "highway.area": HighwayArea}[
            tag
        ](tag, *args, **kwargs)
    except KeyError:
        return Tag(tag, *args, **kwargs)
