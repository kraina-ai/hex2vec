from typing import List
from typing import Dict
import pandas as pd

# def _keep_dig(char):
#     """Keep only digits in a string."""
#     return char == "." or str.isdigit(char)

# def string_2_float(string):
#     """Filter digits in a string and convert to float."""
#     """ Used for Building Height """
#     return float("".join(filter(_keep_dig, string)))


class Tag:
    def __init__(self, tag: str, filter_values: List[str] = None) -> None:
        if "." in tag:
            self._tag = tag.split(".")[0]
            self._subtag = tag.split(".")[1]
        else:
            self._tag = tag
            self._subtag = None

        # store filter values as set for faster lookup
        self.filter_values = set(filter_values) if filter_values else None

    def __str__(self) -> str:
        return self.tag

    @property
    def osmxtag(
        self,
    ) -> str:
        return self._tag

    @property
    def tag(
        self,
    ) -> str:
        return ".".join([self._tag, self._subtag]) if self._subtag else self._tag

    @property
    def columns(
        self,
    ) -> List[str]:
        if self.filter_values is not None:
            return [f"{self.tag}_{value}" for value in self.filter_values]
        else:
            return [self.tag]

    def extract_osmnx_tag(self, df_raw: pd.DataFrame, simplify: bool) -> pd.DataFrame:
        return df_raw[["osmid", self.tag, "geometry"]] if simplify else df_raw

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


class BuildingArea(Tag):
    def __init__(self, tag: str = "building.area", *args, **kwargs) -> None:
        super().__init__(tag, *args, **kwargs)

    def _create_area_column(self, df):
        # create the area column
        # convert the crs to meters
        df[self.tag] = df["geometry"].to_crs(epsg=3857).area
        return df

    def group_df_by_tag_values(self, df):
        # create the area column
        # convert the crs to meters
        df = self._create_area_column(df)
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


def build_tag(tag: str, *args, **kwargs) -> Tag:
    try:
        return {"building.area": BuildingArea,}[
            tag
        ](tag, *args, **kwargs)
    except KeyError:
        return Tag(tag, *args, **kwargs)
