import datetime
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession, Column
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType


DATA_PATH = "train.csv"
CURRENT_YEAR = datetime.datetime.now().year
SEED = 42
TEST_SIZE = 0.2
TO_SCALE_FEATURES = ["LotArea", "GarageArea", "PoolArea", "1stFlrSF", "2ndFlrSF", "Age"]
OTHER_FEATURES = [
    "OverallQual", "OverallCond", "ExterQual", "BsmtCond", "HeatingQC", "KitchenQual", "HasAllUtils",
    "IsFor1Fam", "HasCentralAir", "WasReconstructed"
]
TARGET_COLUMN = "SalePrice"
FEATURES_FOR_SCALING_COLUMN = "features_for_scaling"
SCALED_FEATURES_COLUMN = "scaled_features"
DISTRICT_COLUMN = "Neighborhood"
DISTRICT_INDEX_COLUMN = "NeighborhoodIndex"
DISTRICT_OHE_FEATURES_COLUMN = "NeighborhoodOHE"
FEATURES_COLUMN = "features"
RF_PREDICTION_COLUMN = "rf_prediction"
LR_PREDICTION_COLUMN = "lr_prediction"


def get_numeric_quality(column_name: str) -> Column:
    return (
        when(col(column_name).startswith("Ex"), 5)
        .when(col(column_name).startswith("Gd"), 4)
        .when(col(column_name).startswith("TA"), 3)
        .when(col(column_name).startswith("Fa"), 2)
        .otherwise(1)
    )


spark = (
    SparkSession.builder
    .appName("House Prices")
    .master("local")
    .getOrCreate()
)

spark_df = (
    spark.read
    .format("csv")
    .option("header", True)
    .load(DATA_PATH)
)

spark_df = spark_df.select(
    col("SalePrice").cast(IntegerType()),
    col("LotArea").cast(IntegerType()),
    col("Utilities"),
    col("Neighborhood"),
    col("BldgType"),
    col("OverallQual").cast(IntegerType()),
    col("OverallCond").cast(IntegerType()),
    col("YearBuilt").cast(IntegerType()),
    col("YearRemodAdd").cast(IntegerType()),
    col("ExterQual"),
    col("BsmtCond"),
    col("HeatingQC"),
    col("CentralAir"),
    col("KitchenQual"),
    col("Fireplaces").cast(IntegerType()),
    col("GarageArea").cast(IntegerType()),
    col("PoolArea").cast(IntegerType()),
    col("1stFlrSF").cast(IntegerType()),
    col("2ndFlrSF").cast(IntegerType()),
)

spark_df = (
    spark_df
    .withColumn("HasAllUtils", col("Utilities").startswith("AllPub"))
    .withColumn(
        "IsFor1Fam",
        (
            when(col("BldgType").startswith("1Fam"), True)
            .when(col("BldgType").startswith("TwnhsE"), True)
            .when(col("BldgType").startswith("TwnhsI"), True)
            .otherwise(False)
        )
    )
)

spark_df = (
    spark_df
    .withColumn("ExterQual", get_numeric_quality("ExterQual"))
    .withColumn("BsmtCond", get_numeric_quality("BsmtCond"))
    .withColumn("HeatingQC", get_numeric_quality("HeatingQC"))
    .withColumn("KitchenQual", get_numeric_quality("KitchenQual"))
    .withColumn("HasCentralAir", col("CentralAir").startswith("Y"))
)

spark_df = (
    spark_df
    .withColumn(
        "WasReconstructed", col("YearRemodAdd") != col("YearBuilt")
    )
    .withColumn(
        "Age", col("YearBuilt") * -1 + CURRENT_YEAR
    )
)

# # Random Splitting
# train_spark_df, test_spark_df = spark_df.randomSplit(weights=[1-TEST_SIZE, TEST_SIZE], seed=SEED)

# Scaling
assembler_for_scaling_features = VectorAssembler(
    inputCols=TO_SCALE_FEATURES, outputCol=FEATURES_FOR_SCALING_COLUMN
)
spark_df = assembler_for_scaling_features.transform(spark_df)

scaler = StandardScaler(
    inputCol=FEATURES_FOR_SCALING_COLUMN, outputCol=SCALED_FEATURES_COLUMN,
    withMean=True, withStd=True
)
scaler_model = scaler.fit(spark_df)
spark_df = scaler_model.transform(spark_df)

# OHE district
district_indexer = StringIndexer(
    inputCol=DISTRICT_COLUMN, outputCol=DISTRICT_INDEX_COLUMN
)
district_indexer_model = district_indexer.fit(spark_df)
spark_df = district_indexer_model.transform(spark_df)

district_ohe = OneHotEncoder(
    inputCol=DISTRICT_INDEX_COLUMN, outputCol=DISTRICT_OHE_FEATURES_COLUMN
)
district_ohe_model = district_ohe.fit(spark_df)
spark_df = district_ohe_model.transform(spark_df)

# UNION
union_assembler = VectorAssembler(
    inputCols=[SCALED_FEATURES_COLUMN, DISTRICT_OHE_FEATURES_COLUMN] + OTHER_FEATURES,
    outputCol=FEATURES_COLUMN
)
spark_df = union_assembler.transform(spark_df)


print(spark_df.select(*TO_SCALE_FEATURES, *OTHER_FEATURES, TARGET_COLUMN).show(5, truncate=False))


lr = LinearRegression(featuresCol=FEATURES_COLUMN, labelCol=TARGET_COLUMN, predictionCol=LR_PREDICTION_COLUMN)
lr_model = lr.fit(spark_df)
spark_df = lr_model.transform(spark_df)

rf = RandomForestRegressor(featuresCol=FEATURES_COLUMN, labelCol=TARGET_COLUMN, predictionCol=RF_PREDICTION_COLUMN)
rf_model = rf.fit(spark_df)
spark_df = rf_model.transform(spark_df)

lr_value = RegressionEvaluator(labelCol=TARGET_COLUMN, predictionCol=LR_PREDICTION_COLUMN, metricName="rmse").evaluate(spark_df)
rf_value = RegressionEvaluator(labelCol=TARGET_COLUMN, predictionCol=RF_PREDICTION_COLUMN, metricName="rmse").evaluate(spark_df)


print("Linear Regression - rMSE:", lr_value)
print("Random ForestRegressor Regression - rMSE:", rf_value)
