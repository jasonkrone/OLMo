CREATE EXTERNAL TABLE IF NOT EXISTS temp_lucas.s2orc_oa_2023_01_03 (
    id INT,
    text STRING,
    lang STRING,
    cnt INT,
    freq STRING
)
STORED AS PARQUET
LOCATION 's3://ai2-s2-lucas/s2orc_llm/2023_01_03/stats/s2orc'
tblproperties ("parquet.compression"="SNAPPY");
