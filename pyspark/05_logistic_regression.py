"""
PySpark Script: 05_logistic_regression.py
Purpose: Build a logistic regression model for loan default prediction
Equivalent SAS Program: sas/05_logistic_regression.sas
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, coalesce, count, mean, udf
)
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import (
    VectorAssembler, StringIndexerModel, OneHotEncoder
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("HomeEquity_LogisticRegression") \
    .master("local[*]") \
    .getOrCreate()

# Load and prepare data (replicate cleaning from scripts 02/04)
df = spark.read.csv("data/home_equity.csv", header=True, inferSchema=True)
df = df \
    .withColumn(
        "LTV",
        when(
            (col("VALUE").isNotNull()) & (col("MORTDUE").isNotNull()) & (col("VALUE") > 0),
            col("MORTDUE") / col("VALUE")
        )
    ) \
    .filter(
        col("LOAN").isNotNull() & col("VALUE").isNotNull() & col("BAD").isNotNull() &
        (col("LOAN") > 0) & (col("VALUE") > 0)
    )

# ------------------------------------------------------------------
# Step 1: Prepare modeling dataset - remove records with excessive missing
# SAS equivalent:
#   data work.model_data;
#       set work.home_equity_risk;
#       if LOAN ne . and MORTDUE ne . and VALUE ne .
#          and DEBTINC ne . and DELINQ ne . and CLAGE ne .;
#   run;
# ------------------------------------------------------------------
# The complete-case filter covers only the six predictors listed in the SAS
# DATA step. DEROG, NINQ, JOB and REASON are still used as model predictors
# (see the MODEL/CLASS statements in the SAS program) but are deliberately not
# part of the complete-case filter, so records with missing values in those
# four columns stay in the modeling population.
modelData = df.filter(
    col("LOAN").isNotNull() &
    col("MORTDUE").isNotNull() &
    col("VALUE").isNotNull() &
    col("DEBTINC").isNotNull() &
    col("DELINQ").isNotNull() &
    col("CLAGE").isNotNull()
)

# Cast BAD to double for ML
modelData = modelData.withColumn("label", col("BAD").cast("double"))

# Because JOB and REASON are no longer part of the complete-case filter they
# can be null. Nulls are mapped to an explicit "(Missing)" category so they
# form their own level instead of being merged with the reference level.
MISSING_CATEGORY = "(Missing)"
modelData = modelData \
    .withColumn("JOB_CAT", coalesce(col("JOB"), lit(MISSING_CATEGORY))) \
    .withColumn("REASON_CAT", coalesce(col("REASON"), lit(MISSING_CATEGORY)))

print(f"Modeling dataset size: {modelData.count()} rows")

# ------------------------------------------------------------------
# Step 2: Split into training (70%) and validation (30%)
# SAS equivalent:
#   proc surveyselect data=work.model_data
#       out=work.model_split method=srs samprate=0.7 seed=42;
#   run;
# ------------------------------------------------------------------
train, valid = modelData.randomSplit([0.7, 0.3], seed=42)
print(f"Training set: {train.count()} rows")
print(f"Validation set: {valid.count()} rows")

# ------------------------------------------------------------------
# Step 3: Build ML pipeline
# SAS equivalent:
#   proc logistic data=work.train descending;
#       class JOB(ref='Other') REASON(ref='HomeImp') / param=ref;
#       model BAD = LOAN MORTDUE VALUE DEBTINC DELINQ DEROG CLAGE NINQ
#                   JOB REASON
#                 / selection=stepwise;
#       output out=work.train_scored predicted=pred_prob;
#   run;
#
# PySpark uses a Pipeline with StringIndexer, OneHotEncoder,
# VectorAssembler, and LogisticRegression.
# ------------------------------------------------------------------

# Index categorical variables (equivalent to CLASS statement)
#
# SAS pins the reference levels explicitly: class JOB(ref='Other')
# REASON(ref='HomeImp') / param=ref. In PySpark the reference level is the
# category dropped by OneHotEncoder, which is always the LAST index, and the
# default StringIndexer ordering (descending frequency) is data-driven - so
# 'Other'/'HomeImp' would not reliably end up as the reference.
#
# To pin the reference level we build the StringIndexerModel from an explicit
# label list that omits the reference category, and keep handleInvalid="keep".
# The reference category therefore falls into the indexer's trailing
# "__unknown" bucket, which is exactly the index OneHotEncoder drops - giving
# the SAS reference level. Limitation: any category unseen when the label list
# was built (there should be none, as labels come from the full modeling
# population) is also folded into the reference level.
def refLastLabels(data, inputCol, refCategory):
    """Ordered labels for `inputCol` (descending frequency) minus refCategory."""
    rows = data.groupBy(inputCol).count() \
        .orderBy(col("count").desc(), col(inputCol).asc()) \
        .collect()
    return [r[inputCol] for r in rows if r[inputCol] != refCategory]


JOB_REF = "Other"          # SAS: class JOB(ref='Other')
REASON_REF = "HomeImp"     # SAS: class REASON(ref='HomeImp')

jobLabels = refLastLabels(modelData, "JOB_CAT", JOB_REF)
reasonLabels = refLastLabels(modelData, "REASON_CAT", REASON_REF)

jobIndexer = StringIndexerModel.from_labels(
    jobLabels, inputCol="JOB_CAT", outputCol="JOB_IDX", handleInvalid="keep"
)
reasonIndexer = StringIndexerModel.from_labels(
    reasonLabels, inputCol="REASON_CAT", outputCol="REASON_IDX",
    handleInvalid="keep"
)

print(f"JOB levels (reference '{JOB_REF}' dropped): {jobLabels}")
print(f"REASON levels (reference '{REASON_REF}' dropped): {reasonLabels}")

# One-hot encode categorical variables (equivalent to param=ref)
jobEncoder = OneHotEncoder(
    inputCol="JOB_IDX", outputCol="JOB_VEC"
)
reasonEncoder = OneHotEncoder(
    inputCol="REASON_IDX", outputCol="REASON_VEC"
)

# Numeric feature columns
numericFeatures = [
    "LOAN", "MORTDUE", "VALUE", "DEBTINC",
    "DELINQ", "DEROG", "CLAGE", "NINQ"
]

# Assemble all features into a single vector
# handleInvalid="skip" drops rows with a missing numeric predictor (DEROG or
# NINQ, which are not part of the complete-case filter), mirroring PROC
# LOGISTIC, which deletes observations with missing MODEL variables.
assembler = VectorAssembler(
    inputCols=numericFeatures + ["JOB_VEC", "REASON_VEC"],
    outputCol="features",
    handleInvalid="skip"
)

# Logistic regression model
# SAS equivalent: PROC LOGISTIC with selection=stepwise
# Note: PySpark's LogisticRegression uses elasticNet for regularization
# which provides built-in feature selection similar to stepwise
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8,  # L1 ratio for feature sparsity (like stepwise)
    threshold=0.5
)

# Build the pipeline
pipeline = Pipeline(stages=[
    jobIndexer, reasonIndexer,
    jobEncoder, reasonEncoder,
    assembler, lr
])

# ------------------------------------------------------------------
# Step 4: Train the model
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training Logistic Regression Model")
print("=" * 60)

model = pipeline.fit(train)

# Extract the logistic regression model from the pipeline
lrModel = model.stages[-1]

# Display model coefficients
print(f"\nIntercept: {lrModel.intercept:.4f}")
print(f"Number of features: {len(lrModel.coefficients)}")
print(f"\nCoefficients (non-zero):")
featureNames = numericFeatures + ["JOB_VEC", "REASON_VEC"]
for i, coef in enumerate(lrModel.coefficients):
    if abs(coef) > 0.0001:
        print(f"  Feature {i}: {coef:.6f}")

# ------------------------------------------------------------------
# Step 5: Score the validation dataset
# SAS equivalent:
#   proc plm restore=work.logit_model;
#       score data=work.valid out=work.valid_scored predicted=pred_prob;
#   run;
# ------------------------------------------------------------------
predictions = model.transform(valid)

# ------------------------------------------------------------------
# Step 6: Create confusion matrix
# SAS equivalent:
#   data work.valid_scored;
#       if pred_prob >= 0.5 then PREDICTED_BAD = 1;
#       else PREDICTED_BAD = 0;
#   run;
#   proc freq data=work.valid_scored;
#       tables BAD * PREDICTED_BAD;
#   run;
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Confusion Matrix - Validation Set")
print("(equivalent to PROC FREQ tables BAD * PREDICTED_BAD)")
print("=" * 60)

predictions.groupBy("label", "prediction") \
    .count() \
    .orderBy("label", "prediction") \
    .show()

# ------------------------------------------------------------------
# Step 7: Calculate model performance metrics
# SAS equivalent:
#   proc logistic ... ;
#       roc;
#       ods output Association=work.association_stats;
#   run;
# ------------------------------------------------------------------
print("=" * 60)
print("Model Performance Metrics")
print("(equivalent to PROC LOGISTIC concordance/AUC)")
print("=" * 60)

# AUC - Area Under ROC Curve
# SAS equivalent: c statistic / concordance
binaryEval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = binaryEval.evaluate(predictions)
print(f"\nAUC (Area Under ROC): {auc:.4f}")

# Area Under PR Curve
binaryEvalPr = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)
aupr = binaryEvalPr.evaluate(predictions)
print(f"AUPR (Area Under PR Curve): {aupr:.4f}")

# Accuracy, Precision, Recall, F1
multiEval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

for metricName in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
    multiEval.setMetricName(metricName)
    value = multiEval.evaluate(predictions)
    print(f"{metricName}: {value:.4f}")

# ------------------------------------------------------------------
# Step 8: Score distribution by actual outcome
# SAS equivalent:
#   proc means data=work.valid_scored n mean std min p25 median p75 max;
#       class BAD;
#       var pred_prob;
#   run;
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Predicted Probability Distribution by Actual Outcome")
print("=" * 60)

# Extract probability of default (class 1)
extractProb = udf(lambda v: float(v[1]), DoubleType())
predictions = predictions.withColumn("pred_prob", extractProb(col("probability")))

predictions.groupBy("label") \
    .agg(
        count("pred_prob").alias("n"),
        mean("pred_prob").alias("mean_pred_prob")
    ) \
    .orderBy("label") \
    .show()

# Detailed statistics
# describe() reports n, mean, std, min and max; PROC MEANS also reports p25,
# median and p75, so the quartiles are computed with approxQuantile.
for labelVal in [0.0, 1.0]:
    subset = predictions.filter(col("label") == labelVal)
    print(f"\nActual BAD = {int(labelVal)}:")
    subset.select("pred_prob").describe().show()

    p25, median, p75 = subset.approxQuantile(
        "pred_prob", [0.25, 0.5, 0.75], 0.001
    )
    print(f"  p25:    {p25:.6f}")
    print(f"  median: {median:.6f}")
    print(f"  p75:    {p75:.6f}")

# ------------------------------------------------------------------
# Step 9: Persist output artifacts
# SAS equivalent:
#   store work.logit_model;                  -> saved pipeline model
#   output out=work.valid_scored ...;        -> scored validation predictions
# ------------------------------------------------------------------
MODEL_PATH = "output/logit_model"
PREDICTIONS_PATH = "output/valid_scored"

model.write().overwrite().save(MODEL_PATH)
print(f"\nSaved fitted pipeline model to {MODEL_PATH}")

predictions.select("label", "prediction", "pred_prob") \
    .write.mode("overwrite").csv(PREDICTIONS_PATH, header=True)
print(f"Saved scored validation predictions to {PREDICTIONS_PATH}")

# Clean up
spark.stop()
