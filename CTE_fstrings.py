attributes_cte_fstring = "WITH attrs AS (SELECT DISTINCT {attribute} FROM {table_name})"
query_func_cte_fstring = "COALESCE({func}(CASE WHEN {cond} THEN {measure} END), 0) AS query_{func}_{measure}"
ref_func_cte_fstring = 'COALESCE({func}(CASE WHEN {cond} THEN {measure} END), 0) AS ref_{func}_{measure}'
combined_query_cte_fstring = """
{attributes_cte}
SELECT {all_selections}
FROM attrs a
LEFT JOIN {table_name} ON a.{attribute} = {table_name}.{attribute}
AND id >= {start} AND id < {end}
GROUP BY a.{attribute}
ORDER BY a.{attribute};
"""
make_combined_view_query_fstring = """
SELECT {attribute}, {selections}
FROM {table_name}
WHERE {attribute} IS NOT NULL AND id >= {start} AND id < {end}
GROUP BY {attribute}
ORDER BY {attribute};
"""
