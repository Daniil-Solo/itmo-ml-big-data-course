import datetime
from typing import Literal

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession, Column, DataFrame
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline

# Constants for data and models
DATA_PATH = "train.csv"
TEST_SIZE = 0.2
CURRENT_YEAR = datetime.datetime.now().year
SEED = 42

# Column names
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
    """
    Маппит строковую оценку в числовую
    :param column_name: название столбца для преобразования
    :return: столбец преобразования
    """
    return (
        when(col(column_name).startswith("Ex"), 5)
        .when(col(column_name).startswith("Gd"), 4)
        .when(col(column_name).startswith("TA"), 3)
        .when(col(column_name).startswith("Fa"), 2)
        .otherwise(1)
    )


def select_fields(spark_dataframe: DataFrame) -> DataFrame:
    """
    Выбирает необходимые поля для последующей трансформации и обучения модели.
    Делает преобразование типов
    :param spark_dataframe: датафрейм со всеми полями
    :return: датафрейм с только нужными полями
    """
    return spark_dataframe.select(
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


def transform_features(spark_dataframe: DataFrame) -> DataFrame:
    """
    Преобразует признаки.
    :param spark_dataframe: датафрейм с выбранными полями
    :return: датафрейм с преобразованными признаками
    """
    return (
        spark_dataframe
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
        .withColumn("ExterQual", get_numeric_quality("ExterQual"))
        .withColumn("BsmtCond", get_numeric_quality("BsmtCond"))
        .withColumn("HeatingQC", get_numeric_quality("HeatingQC"))
        .withColumn("KitchenQual", get_numeric_quality("KitchenQual"))
        .withColumn("HasCentralAir", col("CentralAir").startswith("Y"))
        .withColumn("WasReconstructed", col("YearRemodAdd") != col("YearBuilt"))
        .withColumn("Age", CURRENT_YEAR - col("YearBuilt"))
    )


def create_pipeline() -> Pipeline:
    """
    Создает пайплайн для подготовки и обучения модели.
    :return: пайплайн
    """
    # Нормализация числовых признаков
    assembler_for_scaling_features = VectorAssembler(
        inputCols=TO_SCALE_FEATURES, outputCol=FEATURES_FOR_SCALING_COLUMN
    )
    scaler = StandardScaler(
        inputCol=FEATURES_FOR_SCALING_COLUMN, outputCol=SCALED_FEATURES_COLUMN,
        withMean=True, withStd=True
    )
    # OneHot-кодирование для района
    district_indexer = StringIndexer(
        inputCol=DISTRICT_COLUMN, outputCol=DISTRICT_INDEX_COLUMN
    )
    district_ohe = OneHotEncoder(
        inputCol=DISTRICT_INDEX_COLUMN, outputCol=DISTRICT_OHE_FEATURES_COLUMN
    )
    # Объединение всех фичей
    union_assembler = VectorAssembler(
        inputCols=[SCALED_FEATURES_COLUMN, DISTRICT_OHE_FEATURES_COLUMN] + OTHER_FEATURES,
        outputCol=FEATURES_COLUMN
    )
    # Модель случайного леса
    rf_model = RandomForestRegressor(
        featuresCol=FEATURES_COLUMN, labelCol=TARGET_COLUMN,
        predictionCol=RF_PREDICTION_COLUMN, seed=SEED
    )
    # Сам пайплайн
    return Pipeline(
        stages=[
            assembler_for_scaling_features,
            scaler,
            district_indexer,
            district_ohe,
            union_assembler,
            rf_model
        ]
    )


def evaluate_pipeline(
        pipeline: Pipeline, train_dataframe: DataFrame, test_dataframe: DataFrame,
        metric_name: Literal["rmse", "mse", "r2", "mae", "var"] = "rmse"
) -> tuple[float, float]:
    """
    Обучает и оценивает пайплайн по метрике качества.
    :param metric_name: название метрики: rmse, mse, r2, mae, var
    :param pipeline: пайплайн обработки данных
    :param train_dataframe: тренировочный датафрейм
    :param test_dataframe: тестовый датафрейм
    :return: значения метрики на трейне и на тесте
    """
    pipeline_model = pipeline.fit(train_dataframe)
    train_predictions_df = pipeline_model.transform(train_dataframe)
    test_predictions_df = pipeline_model.transform(test_dataframe)
    train_metric = (
        RegressionEvaluator(labelCol=TARGET_COLUMN, predictionCol=RF_PREDICTION_COLUMN, metricName=metric_name)
        .evaluate(train_predictions_df)
    )
    test_metric = (
        RegressionEvaluator(labelCol=TARGET_COLUMN, predictionCol=RF_PREDICTION_COLUMN, metricName=metric_name)
        .evaluate(test_predictions_df)
    )
    return train_metric, test_metric


# Создание Spark-сессии
spark = (
    SparkSession.builder
    .appName("House Prices")
    .master("local")
    .getOrCreate()
)

# Работа с датафреймом
spark_df = spark.read.format("csv").option("header", True).load(DATA_PATH)
spark_df = select_fields(spark_df)
spark_df = transform_features(spark_df)
spark_df.select(*TO_SCALE_FEATURES, *OTHER_FEATURES, TARGET_COLUMN).show(5, truncate=False)

# Обучение пайплайна обработки данных
train_spark_df, test_spark_df = spark_df.randomSplit(weights=[1 - TEST_SIZE, TEST_SIZE], seed=SEED)
train_value, test_value = evaluate_pipeline(create_pipeline(), train_spark_df, test_spark_df, "rmse")
print(f"Random ForestRegressor - rMSE on train: {round(train_value, 1)}")
print(f"Random ForestRegressor - rMSE on test: {round(test_value, 1)}")
