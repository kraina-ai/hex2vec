# import pandas as pd

# def _keep_dig(char):
#     """Keep only digits in a string."""
#     return char == "." or str.isdigit(char)

# def string_2_float(string):
#     """Filter digits in a string and convert to float."""
#     """ Used for Building Height """
#     return float("".join(filter(_keep_dig, string)))


# PROCESSING = {
#     'building.height': string_2_float,
#     'building.levels': string_2_float
# }


# # lower case 
# class Tag:

#     def __init__(self, tag: str) -> None:
#         if '.' in tag:
#             self._tag = tag.split('.')[0]
#             self._subtag = tag.split('.')[1]
#         else:
#             self._tag = tag
#             self._subtag = None

#     def __str__(self) -> str:
#         return self.tag

#     @property
#     def osmxtag(self, ) -> str:
#         return self._tag
    
#     @property
#     def tag(self, ) -> str:
#         return ".".join([self._tag, self._subtag]) if self._subtag else self._tag

#     def extract_osmnx_tag(self, df_raw: pd.DataFrame, simplify: bool) -> pd.DataFrame:
#         if self.tag in PROCESSING:
#             df_raw[self.tag] = df_raw[self.tag].apply(PROCESSING[self.tag])
#         return df_raw[['osmid', self.tag, 'geometry']] if simplify else df_raw

#     def group_df_by_tag_values(self, df, ):
#         """""""
#         tmp = df.reset_index(drop=True)[["h3", tag]].copy(deep=True)
#         indicators = pd.get_dummies(tmp, columns=[tag], prefix=tag).groupby("h3").sum().astype("int16")
#         return indicators.reset_index()

    