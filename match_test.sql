WITH vp AS (
  SELECT vp.*
  FROM victim_pairs vp
  WHERE (victim_row_id_l = '100004075:0:0' AND victim_row_id_r = '100010689:0:0')
     OR (victim_row_id_l = '100010689:0:0' AND victim_row_id_r = '100004075:0:0')
),
pair_levels AS (
  SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'victim_forename_norm_victim_surname_norm' AS comparison_name,
         gamma_victim_forename_norm_victim_surname_norm AS comparison_vector_value
  FROM vp
  UNION ALL SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'date_proximity', gamma_date_proximity FROM vp
  UNION ALL SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'age_proximity', gamma_age_proximity FROM vp
  UNION ALL SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'lat_lon', gamma_lat_lon FROM vp                -- use 'distance_in_km' if that's what you have
  UNION ALL SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'victim_sex', gamma_victim_sex FROM vp
  UNION ALL SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'offender_name_norm', gamma_offender_name_norm FROM vp
  UNION ALL SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'weapon', gamma_weapon FROM vp
  UNION ALL SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'circumstance', gamma_circumstance FROM vp
  UNION ALL SELECT victim_row_id_l, victim_row_id_r, match_probability, match_weight,
         'same_article_penalty', gamma_same_article_penalty FROM vp
)
SELECT
  pl.match_probability,
  pl.match_weight,
  mw.comparison_sort_order,
  pl.comparison_name,
  pl.comparison_vector_value,
  mw.label_for_charts,
  mw.m_probability,
  mw.u_probability,
  mw.log2_bayes_factor AS contribution
FROM pair_levels pl
JOIN victim_pairs vp USING (victim_row_id_l, victim_row_id_r)
LEFT JOIN match_weights mw
  ON mw.comparison_name = pl.comparison_name
 AND mw.comparison_vector_value = pl.comparison_vector_value
ORDER BY mw.comparison_sort_order, pl.comparison_name;