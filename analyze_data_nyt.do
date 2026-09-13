* New York Times / New York City Stata analysis entry point.
* One row in the input is one SHR entry in the NYC five-borough scope.

cd "\\apporto.com\dfs\STNFRD\Users\s9130_stnfrd\Documents\NYC"

do ../DC/analyze_data_wp.do ///
    "homicide_data2_monadic.csv" ///
    "." ///
    1981 ///
    2000 ///
    "../DC/category_group_crosswalk.csv" ///
    1982 ///
    0.25
