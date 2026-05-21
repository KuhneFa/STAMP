# STAMP Monitoring Dashboard

This is a small Streamlit dashboard for looking at prediction CSVs from a STAMP-style pathology workflow. I built it as a practical monitoring layer: after a model has produced patient-level predictions, the dashboard helps inspect confidence, uncertain cases, cohort differences, and basic performance when labels are available.

It is intentionally lightweight. The aim is not to replace a validation study, but to make the usual deployment questions visible:

- Are many cases close to the decision boundary?
- Which cases should be sent to manual review?
- Do scores look different for an external cohort or another scanner?
- If labels arrive later, has performance changed?

This is a research prototype. It is not clinically validated and should not be used for medical decisions.

## Expected Input

The smallest useful CSV needs two columns:

```csv
patient_id,prediction_score
```

Optional columns:

```csv
slide_id,cohort,center,scanner,date,predicted_label,true_label
```

The loader also handles STAMP-like `patient-preds.csv` files, for example:

```csv
PATIENT,KRAS,pred,KRAS_mutated,KRAS_wild type,loss
```

If `prediction_score` is missing, the app looks for probability-like columns between 0 and 1 and uses the largest value as a simple confidence score.

## Run

Install the small dashboard dependency set:

```bash
pip install -r examples/monitoring_dashboard/requirements.txt
```

From the repository root:

```bash
streamlit run examples/monitoring_dashboard/monitoring_dashboard.py
```

Or, if you are already inside this folder:

```bash
streamlit run monitoring_dashboard.py
```

Without uploading a file, the dashboard opens with `sample_predictions.csv`.

## Using STAMP Outputs

After running STAMP cross-validation or deployment, upload a file such as:

```text
stamp-test-experiment/crossval/split-0/patient-preds.csv
```

For richer monitoring, add metadata columns such as `cohort`, `center`, `scanner`,
or `date` to the prediction CSV. The dashboard will use whichever of these columns
are present.

## CAMELYON16 Demo Data

For a slightly more realistic demo, generate a CSV from the public
`torchmil/Camelyon16_MIL` dataset:

```bash
python examples/monitoring_dashboard/make_camelyon16_monitoring_sample.py
```

This downloads only the small split and label files, not the large WSI or feature
archives. The generated file is:

```text
examples/monitoring_dashboard/camelyon16_monitoring_predictions.csv
```

Only the small split and label files are downloaded. The large WSI and feature
archives are not downloaded.

The slide IDs and labels come from CAMELYON16-MIL. The prediction scores, centers,
scanners, and dates are simulated so the dashboard has something interesting to
show. This file is useful for testing the monitoring workflow, but it is not a
model benchmark and should not be reported as model performance.

## What To Look At

1. **Dataset overview**: number of patients, slides, cohorts, centers, and mean score.
2. **Prediction and uncertainty**: scores near 0.5 are treated as more uncertain.
3. **Drift analysis**: compare score distributions across cohort, center, scanner, or month.
4. **Performance monitoring**: shown only when `true_label` is available.
5. **Flagged cases**: cases above the review threshold can be exported as CSV.

## How The Controls Work

- `Review threshold`: cases with uncertainty above this value are flagged for review.
- `Drift mean-shift warning`: warns when a group's mean score differs strongly from the reference group.
- `KS p-value warning`: warns when a Kolmogorov-Smirnov test suggests that a group's score distribution differs from the reference distribution.

The uncertainty estimate is deliberately simple:

```python
uncertainty = 1 - abs(prediction_score - 0.5) * 2
```

That means predictions near 0.5 are treated as uncertain, while predictions near
0 or 1 are treated as more confident. This is only a monitoring heuristic, not a
validated uncertainty method.
