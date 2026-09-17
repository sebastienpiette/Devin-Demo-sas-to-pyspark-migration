"""
Test Suite: test_pyspark_outputs.py
Purpose: Validate PySpark transformations against expected outputs
Ensures correctness of the migrated SAS-to-PySpark logic.
"""

import unittest
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, mean, count


class TestHomeEquityPySpark(unittest.TestCase):
    """Test suite for HOME_EQUITY PySpark migration scripts."""

    @classmethod
    def setUpClass(cls):
        """Create a local SparkSession and load the dataset."""
        cls.spark = SparkSession.builder \
            .appName("HomeEquity_Tests") \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .getOrCreate()

        # Determine the data path relative to the project root
        testDir = os.path.dirname(os.path.abspath(__file__))
        projectRoot = os.path.dirname(testDir)
        dataPath = os.path.join(projectRoot, "data", "home_equity.csv")

        cls.df = cls.spark.read.csv(dataPath, header=True, inferSchema=True)

    @classmethod
    def tearDownClass(cls):
        """Stop the SparkSession."""
        cls.spark.stop()

    # ------------------------------------------------------------------
    # Test 01: Data Loading
    # ------------------------------------------------------------------
    def test_data_loads_successfully(self):
        """Verify the CSV loads without errors."""
        self.assertIsNotNone(self.df)
        self.assertTrue(self.df.count() > 0)

    def test_row_count(self):
        """Verify expected row count from home_equity.csv."""
        rowCount = self.df.count()
        self.assertEqual(rowCount, 5960, f"Expected 5960 rows, got {rowCount}")

    def test_expected_columns_exist(self):
        """Verify all expected columns are present."""
        expectedColumns = [
            "BAD", "LOAN", "MORTDUE", "VALUE", "REASON", "JOB",
            "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO",
            "DEBTINC", "APPDATE", "CITY", "STATE", "DIVISION", "REGION"
        ]
        actualColumns = self.df.columns
        for colName in expectedColumns:
            self.assertIn(
                colName, actualColumns,
                f"Missing expected column: {colName}"
            )

    def test_column_count(self):
        """Verify the number of columns."""
        self.assertEqual(len(self.df.columns), 18)

    # ------------------------------------------------------------------
    # Test 02: Data Cleaning - Derived Columns
    # ------------------------------------------------------------------
    def test_ltv_calculation(self):
        """Verify LTV (Loan-to-Value) is computed correctly."""
        dfClean = self.df.withColumn(
            "LTV",
            when(
                (col("VALUE").isNotNull()) &
                (col("MORTDUE").isNotNull()) &
                (col("VALUE") > 0),
                col("MORTDUE") / col("VALUE")
            )
        )
        self.assertIn("LTV", dfClean.columns)

        # LTV should be positive where computed
        validLtv = dfClean.filter(col("LTV").isNotNull())
        self.assertTrue(validLtv.count() > 0, "No valid LTV values computed")

        # Check a known calculation: row where MORTDUE=25860, VALUE=39025
        # LTV should be approximately 0.6627
        sample = dfClean.filter(
            (col("MORTDUE") == 25860) & (col("VALUE") == 39025)
        ).select("LTV").collect()
        if len(sample) > 0:
            ltv = sample[0]["LTV"]
            self.assertAlmostEqual(ltv, 25860 / 39025, places=4)

    def test_loan_outcome_derivation(self):
        """Verify LOAN_OUTCOME is derived correctly from BAD."""
        dfClean = self.df.withColumn(
            "LOAN_OUTCOME",
            when(col("BAD") == 0, lit("Paid"))
            .when(col("BAD") == 1, lit("Default"))
        )
        self.assertIn("LOAN_OUTCOME", dfClean.columns)

        # All BAD=0 should map to 'Paid'
        paidCount = dfClean.filter(
            (col("BAD") == 0) & (col("LOAN_OUTCOME") == "Paid")
        ).count()
        totalPaid = self.df.filter(col("BAD") == 0).count()
        self.assertEqual(paidCount, totalPaid)

        # All BAD=1 should map to 'Default'
        defaultCount = dfClean.filter(
            (col("BAD") == 1) & (col("LOAN_OUTCOME") == "Default")
        ).count()
        totalDefault = self.df.filter(col("BAD") == 1).count()
        self.assertEqual(defaultCount, totalDefault)

    def test_filtering_removes_nulls(self):
        """Verify filtering removes records with null critical fields."""
        dfFiltered = self.df.filter(
            col("LOAN").isNotNull() &
            col("VALUE").isNotNull() &
            col("BAD").isNotNull()
        )
        # Should have fewer or equal rows
        self.assertLessEqual(dfFiltered.count(), self.df.count())
        # No nulls in critical columns
        self.assertEqual(
            dfFiltered.filter(col("LOAN").isNull()).count(), 0
        )
        self.assertEqual(
            dfFiltered.filter(col("VALUE").isNull()).count(), 0
        )
        self.assertEqual(
            dfFiltered.filter(col("BAD").isNull()).count(), 0
        )

    # ------------------------------------------------------------------
    # Test 03: Aggregation and Reporting
    # ------------------------------------------------------------------
    def test_frequency_counts(self):
        """Verify groupBy counts produce valid results."""
        jobCounts = self.df.groupBy("JOB").count().collect()
        self.assertTrue(len(jobCounts) > 0, "No job categories found")

        # Check that total counts sum to dataset size (minus nulls);
        # groupBy emits a null group, which is excluded here.
        totalFromGroups = sum(
            row["count"] for row in jobCounts if row["JOB"] is not None
        )
        nonNullJobs = self.df.filter(col("JOB").isNotNull()).count()
        self.assertEqual(totalFromGroups, nonNullJobs)

    def test_summary_stats_by_group(self):
        """Verify grouped summary statistics are computed."""
        dfClean = self.df.withColumn(
            "LOAN_OUTCOME",
            when(col("BAD") == 0, lit("Paid"))
            .when(col("BAD") == 1, lit("Default"))
        )

        stats = dfClean.groupBy("LOAN_OUTCOME").agg(
            count("LOAN").alias("N"),
            mean("LOAN").alias("Mean_LOAN")
        ).collect()

        self.assertTrue(len(stats) > 0)
        for row in stats:
            if row["LOAN_OUTCOME"] is not None:
                self.assertTrue(row["N"] > 0)
                self.assertTrue(row["Mean_LOAN"] > 0)

    def test_sql_query_top_states(self):
        """Verify Spark SQL query returns valid results."""
        self.df.createOrReplaceTempView("home_equity_test")

        result = self.spark.sql("""
            SELECT STATE, COUNT(*) AS num_loans, AVG(LOAN) AS avg_loan
            FROM home_equity_test
            GROUP BY STATE
            HAVING COUNT(*) >= 10
            ORDER BY avg_loan DESC
            LIMIT 10
        """).collect()

        self.assertTrue(len(result) > 0, "SQL query returned no results")
        self.assertLessEqual(len(result), 10)
        # Verify ordering (descending by avg_loan)
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i]["avg_loan"], result[i + 1]["avg_loan"])

    # ------------------------------------------------------------------
    # Test 04: Risk Segmentation
    # ------------------------------------------------------------------
    def test_risk_categories_created(self):
        """Verify risk category columns are computed."""
        dfRisk = self.df.withColumn(
            "LTV",
            when(
                (col("VALUE").isNotNull()) & (col("MORTDUE").isNotNull()) & (col("VALUE") > 0),
                col("MORTDUE") / col("VALUE")
            )
        ).withColumn(
            "LTV_RISK_CAT",
            when(col("LTV").isNull(), lit(None))
            .when(col("LTV") < 0.60, lit("Low"))
            .when(col("LTV") < 0.80, lit("Medium"))
            .otherwise(lit("High"))
        ).withColumn(
            "DTI_RISK_CAT",
            when(col("DEBTINC").isNull(), lit(None))
            .when(col("DEBTINC") < 30, lit("Low"))
            .when(col("DEBTINC") < 40, lit("Medium"))
            .when(col("DEBTINC") < 50, lit("High"))
            .otherwise(lit("Very High"))
        )

        self.assertIn("LTV_RISK_CAT", dfRisk.columns)
        self.assertIn("DTI_RISK_CAT", dfRisk.columns)

        # Verify valid categories
        validLtvCats = {"Low", "Medium", "High", None}
        actualLtvCats = set(
            row["LTV_RISK_CAT"]
            for row in dfRisk.select("LTV_RISK_CAT").distinct().collect()
        )
        self.assertTrue(actualLtvCats.issubset(validLtvCats))

    def test_risk_score_range(self):
        """Verify composite risk score is within expected range (0-10)."""
        dfRisk = self.df.withColumn(
            "LTV",
            when(
                (col("VALUE").isNotNull()) & (col("MORTDUE").isNotNull()) & (col("VALUE") > 0),
                col("MORTDUE") / col("VALUE")
            )
        )

        # LTV component
        ltvScore = when(col("LTV").isNull(), lit(0)) \
            .when(col("LTV") >= 0.80, lit(3)) \
            .when(col("LTV") >= 0.60, lit(1.5)) \
            .otherwise(lit(0))
        dtiScore = when(col("DEBTINC").isNull(), lit(0)) \
            .when(col("DEBTINC") >= 50, lit(3)) \
            .when(col("DEBTINC") >= 40, lit(2)) \
            .when(col("DEBTINC") >= 30, lit(1)) \
            .otherwise(lit(0))
        delinqScore = when(col("DELINQ").isNull(), lit(0)) \
            .when(col("DELINQ") >= 4, lit(2)) \
            .when(col("DELINQ") >= 2, lit(1.5)) \
            .when(col("DELINQ") == 1, lit(0.5)) \
            .otherwise(lit(0))
        derogScore = when(col("DEROG").isNull(), lit(0)) \
            .when(col("DEROG") >= 3, lit(2)) \
            .when(col("DEROG") >= 1, lit(1)) \
            .otherwise(lit(0))

        dfRisk = dfRisk.withColumn(
            "RISK_SCORE",
            ltvScore + dtiScore + delinqScore + derogScore
        )

        # All scores should be between 0 and 10
        minScore = dfRisk.agg({"RISK_SCORE": "min"}).collect()[0][0]
        maxScore = dfRisk.agg({"RISK_SCORE": "max"}).collect()[0][0]
        self.assertGreaterEqual(minScore, 0)
        self.assertLessEqual(maxScore, 10)

    def test_risk_segment_labels(self):
        """Verify risk segment labels are correct."""
        dfRisk = self.df.withColumn(
            "LTV",
            when(
                (col("VALUE").isNotNull()) & (col("MORTDUE").isNotNull()) & (col("VALUE") > 0),
                col("MORTDUE") / col("VALUE")
            )
        )

        ltvScore = when(col("LTV").isNull(), lit(0)) \
            .when(col("LTV") >= 0.80, lit(3)) \
            .when(col("LTV") >= 0.60, lit(1.5)) \
            .otherwise(lit(0))
        dtiScore = when(col("DEBTINC").isNull(), lit(0)) \
            .when(col("DEBTINC") >= 50, lit(3)) \
            .when(col("DEBTINC") >= 40, lit(2)) \
            .when(col("DEBTINC") >= 30, lit(1)) \
            .otherwise(lit(0))
        delinqScore = when(col("DELINQ").isNull(), lit(0)) \
            .when(col("DELINQ") >= 4, lit(2)) \
            .when(col("DELINQ") >= 2, lit(1.5)) \
            .when(col("DELINQ") == 1, lit(0.5)) \
            .otherwise(lit(0))
        derogScore = when(col("DEROG").isNull(), lit(0)) \
            .when(col("DEROG") >= 3, lit(2)) \
            .when(col("DEROG") >= 1, lit(1)) \
            .otherwise(lit(0))

        dfRisk = dfRisk.withColumn(
            "RISK_SCORE",
            ltvScore + dtiScore + delinqScore + derogScore
        ).withColumn(
            "RISK_SEGMENT",
            when(col("RISK_SCORE") < 3, lit("Low Risk"))
            .when(col("RISK_SCORE") < 5, lit("Medium Risk"))
            .when(col("RISK_SCORE") < 7, lit("High Risk"))
            .otherwise(lit("Very High Risk"))
        )

        validSegments = {"Low Risk", "Medium Risk", "High Risk", "Very High Risk"}
        actualSegments = set(
            row["RISK_SEGMENT"]
            for row in dfRisk.select("RISK_SEGMENT").distinct().collect()
        )
        self.assertTrue(actualSegments.issubset(validSegments))

    # ------------------------------------------------------------------
    # Test 05: Logistic Regression
    #
    # The logistic-regression tests below are structural / invariant checks:
    # they validate that the migrated PySpark pipeline reproduces the SAS
    # program's *logic* (modeling population, split, reference categories,
    # confusion-matrix shape, probability ranges and direction).
    #
    # They are NOT numeric parity tests. True parity would require a captured
    # SAS baseline - AUC / c-statistic, model coefficients, confusion-matrix
    # cell counts and per-row predicted probabilities, compared with agreed
    # tolerances. No such SAS golden output exists in this repository, so the
    # assertions here are limited to invariants that must hold regardless of
    # the exact fitted numbers.
    # ------------------------------------------------------------------

    # Modeling population per sas/05_logistic_regression.sas lines 12-15:
    # complete cases on six predictors only. DEROG, NINQ, JOB and REASON are
    # model predictors but are not part of the complete-case filter.
    COMPLETE_CASE_COLUMNS = [
        "LOAN", "MORTDUE", "VALUE", "DEBTINC", "DELINQ", "CLAGE"
    ]
    EXPECTED_MODEL_ROWS = 3897
    MISSING_CATEGORY = "(Missing)"
    JOB_REF = "Other"        # SAS: class JOB(ref='Other')
    REASON_REF = "HomeImp"   # SAS: class REASON(ref='HomeImp')

    def _modelData(self):
        """Modeling dataset matching pyspark/05_logistic_regression.py."""
        from pyspark.sql.functions import coalesce

        condition = col(self.COMPLETE_CASE_COLUMNS[0]).isNotNull()
        for colName in self.COMPLETE_CASE_COLUMNS[1:]:
            condition = condition & col(colName).isNotNull()

        return self.df.filter(condition) \
            .withColumn("label", col("BAD").cast("double")) \
            .withColumn(
                "JOB_CAT", coalesce(col("JOB"), lit(self.MISSING_CATEGORY))
            ) \
            .withColumn(
                "REASON_CAT",
                coalesce(col("REASON"), lit(self.MISSING_CATEGORY))
            )

    def _refLastLabels(self, data, inputCol, refCategory):
        """Labels ordered by descending frequency, reference category removed.

        Mirrors the script: the reference category is left out of the label
        list so that it lands in the indexer's trailing "__unknown" bucket,
        which is the index OneHotEncoder drops.
        """
        rows = data.groupBy(inputCol).count() \
            .orderBy(col("count").desc(), col(inputCol).asc()) \
            .collect()
        return [r[inputCol] for r in rows if r[inputCol] != refCategory]

    def _buildPipeline(self, modelData):
        """Pipeline mirroring pyspark/05_logistic_regression.py."""
        from pyspark.ml.feature import (
            VectorAssembler, StringIndexerModel, OneHotEncoder
        )
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml import Pipeline

        jobIdx = StringIndexerModel.from_labels(
            self._refLastLabels(modelData, "JOB_CAT", self.JOB_REF),
            inputCol="JOB_CAT", outputCol="JOB_IDX", handleInvalid="keep"
        )
        reasonIdx = StringIndexerModel.from_labels(
            self._refLastLabels(modelData, "REASON_CAT", self.REASON_REF),
            inputCol="REASON_CAT", outputCol="REASON_IDX",
            handleInvalid="keep"
        )
        jobEnc = OneHotEncoder(inputCol="JOB_IDX", outputCol="JOB_VEC")
        reasonEnc = OneHotEncoder(
            inputCol="REASON_IDX", outputCol="REASON_VEC"
        )

        assembler = VectorAssembler(
            inputCols=["LOAN", "MORTDUE", "VALUE", "DEBTINC",
                       "DELINQ", "DEROG", "CLAGE", "NINQ",
                       "JOB_VEC", "REASON_VEC"],
            outputCol="features",
            handleInvalid="skip"
        )

        # Hyperparameters must match the shipped script.
        lr = LogisticRegression(
            featuresCol="features", labelCol="label",
            maxIter=100, regParam=0.01, elasticNetParam=0.8, threshold=0.5
        )

        return Pipeline(stages=[
            jobIdx, reasonIdx, jobEnc, reasonEnc, assembler, lr
        ])

    def _scoredValidation(self):
        """Fit on train, score valid; cached across tests."""
        if getattr(type(self), "_cachedScored", None) is None:
            modelData = self._modelData()
            train, valid = modelData.randomSplit([0.7, 0.3], seed=42)
            model = self._buildPipeline(modelData).fit(train)
            predictions = model.transform(valid).cache()
            type(self)._cachedScored = (valid, predictions)
        return type(self)._cachedScored

    def test_modeling_population_size(self):
        """Six-predictor complete-case filter yields a stable row count."""
        modelData = self._modelData()
        rowCount = modelData.count()
        self.assertEqual(
            rowCount, self.EXPECTED_MODEL_ROWS,
            f"Expected {self.EXPECTED_MODEL_ROWS} complete cases, "
            f"got {rowCount}"
        )
        # Deterministic: recomputing the filter gives the same count.
        self.assertEqual(self._modelData().count(), rowCount)

        # DEROG / NINQ / JOB / REASON are NOT part of the filter, so the
        # population still contains records missing those predictors.
        self.assertGreater(
            modelData.filter(
                col("DEROG").isNull() | col("NINQ").isNull() |
                col("JOB").isNull() | col("REASON").isNull()
            ).count(),
            0,
            "Filter should not require DEROG/NINQ/JOB/REASON"
        )

    def test_train_validation_split(self):
        """Split is 70/30, exhaustive and deterministic for a fixed seed."""
        modelData = self._modelData()
        totalCount = modelData.count()

        train, valid = modelData.randomSplit([0.7, 0.3], seed=42)
        trainCount, validCount = train.count(), valid.count()

        self.assertEqual(trainCount + validCount, totalCount)
        self.assertAlmostEqual(trainCount / totalCount, 0.7, delta=0.03)
        self.assertAlmostEqual(validCount / totalCount, 0.3, delta=0.03)

        # Same seed -> same split.
        train2, valid2 = modelData.randomSplit([0.7, 0.3], seed=42)
        self.assertEqual(train2.count(), trainCount)
        self.assertEqual(valid2.count(), validCount)

    def test_categorical_reference_categories(self):
        """JOB/REASON encode with the SAS reference level dropped."""
        from pyspark.ml.feature import StringIndexerModel, OneHotEncoder
        from pyspark.ml import Pipeline

        modelData = self._modelData()
        jobLabels = self._refLastLabels(modelData, "JOB_CAT", self.JOB_REF)
        reasonLabels = self._refLastLabels(
            modelData, "REASON_CAT", self.REASON_REF
        )

        # The reference category is deliberately excluded from the labels.
        self.assertNotIn(self.JOB_REF, jobLabels)
        self.assertNotIn(self.REASON_REF, reasonLabels)
        # Nulls form their own level rather than joining the reference.
        self.assertIn(self.MISSING_CATEGORY, jobLabels)
        self.assertIn(self.MISSING_CATEGORY, reasonLabels)

        encoded = Pipeline(stages=[
            StringIndexerModel.from_labels(
                jobLabels, inputCol="JOB_CAT", outputCol="JOB_IDX",
                handleInvalid="keep"
            ),
            StringIndexerModel.from_labels(
                reasonLabels, inputCol="REASON_CAT", outputCol="REASON_IDX",
                handleInvalid="keep"
            ),
            OneHotEncoder(inputCol="JOB_IDX", outputCol="JOB_VEC"),
            OneHotEncoder(inputCol="REASON_IDX", outputCol="REASON_VEC"),
        ]).fit(modelData).transform(modelData)

        # One dummy column per non-reference level.
        firstRow = encoded.select("JOB_VEC", "REASON_VEC").head()
        self.assertEqual(firstRow["JOB_VEC"].size, len(jobLabels))
        self.assertEqual(firstRow["REASON_VEC"].size, len(reasonLabels))

        # Reference rows encode as the all-zero vector.
        for catCol, vecCol, refValue in [
            ("JOB_CAT", "JOB_VEC", self.JOB_REF),
            ("REASON_CAT", "REASON_VEC", self.REASON_REF),
        ]:
            refRows = encoded.filter(col(catCol) == refValue) \
                .select(vecCol).limit(5).collect()
            self.assertTrue(
                len(refRows) > 0, f"No rows with {catCol} = {refValue}"
            )
            for row in refRows:
                self.assertEqual(
                    row[vecCol].numNonzeros(), 0,
                    f"{refValue} should be the dropped reference level"
                )

    def test_confusion_matrix_cells(self):
        """Confusion matrix has all four cells and sums to the scored rows."""
        valid, predictions = self._scoredValidation()

        matrix = {
            (row["label"], row["prediction"]): row["count"]
            for row in predictions.groupBy("label", "prediction")
            .count().collect()
        }

        for cell in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]:
            self.assertIn(cell, matrix, f"Missing confusion matrix cell {cell}")
            self.assertGreater(matrix[cell], 0)

        scoredCount = predictions.count()
        self.assertEqual(sum(matrix.values()), scoredCount)
        # Rows with a missing numeric predictor are skipped by the assembler,
        # mirroring PROC LOGISTIC dropping incomplete observations.
        self.assertLessEqual(scoredCount, valid.count())

    def test_predicted_probability_range_and_direction(self):
        """Predicted probabilities are in [0, 1] and rank defaults higher."""
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType

        _, predictions = self._scoredValidation()
        extractProb = udf(lambda v: float(v[1]), DoubleType())
        scored = predictions.withColumn(
            "pred_prob", extractProb(col("probability"))
        )

        outOfRange = scored.filter(
            (col("pred_prob") < 0.0) | (col("pred_prob") > 1.0) |
            col("pred_prob").isNull()
        ).count()
        self.assertEqual(outOfRange, 0, "pred_prob must be within [0, 1]")

        means = {
            row["label"]: row["avg_prob"]
            for row in scored.groupBy("label")
            .agg(mean("pred_prob").alias("avg_prob")).collect()
        }
        self.assertIn(0.0, means)
        self.assertIn(1.0, means)
        self.assertGreater(
            means[1.0], means[0.0],
            "Mean predicted probability for BAD=1 should exceed BAD=0"
        )

    def test_logistic_regression_model(self):
        """Verify logistic regression pipeline produces predictions."""
        from pyspark.ml.evaluation import BinaryClassificationEvaluator

        modelData = self._modelData()

        self.assertTrue(
            modelData.count() > 100,
            "Not enough complete cases for modeling"
        )

        train, valid = modelData.randomSplit([0.7, 0.3], seed=42)
        model = self._buildPipeline(modelData).fit(train)
        predictions = model.transform(valid)

        # Verify predictions exist
        self.assertIn("prediction", predictions.columns)
        self.assertIn("probability", predictions.columns)

        # Verify predictions are 0 or 1
        predValues = set(
            row["prediction"]
            for row in predictions.select("prediction").distinct().collect()
        )
        self.assertTrue(predValues.issubset({0.0, 1.0}))

        # Verify AUC is reasonable (> 0.5 = better than random)
        evaluator = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)
        self.assertGreater(auc, 0.5, f"AUC ({auc:.4f}) should be > 0.5")


if __name__ == "__main__":
    unittest.main()
