# EDA Report - Personal Wellbeing Time-Series Monitoring App

## Dataset Overview

- Rows: 3412
- Students: 49
- Date range: 2013-03-25 to 2013-08-16
- Risk rows, wellbeing score < 70: 1686

## Preprocessing Checks

- Missing values: `{}`
- Invalid values: `{}`
- Out-of-range values: `{}`

## Numeric Summary

| Feature | Min | Q1 | Mean | Median | Q3 | Max | Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sleep | 0.0 | 6.4 | 7.166 | 7.1 | 8.0 | 14.0 | 1.537 |
| stress | 1.0 | 2.8 | 3.893 | 3.6 | 5.2 | 10.0 | 1.971 |
| mood | 1.0 | 5.5 | 6.123 | 5.5 | 7.8 | 10.0 | 1.743 |
| energy | 1.8 | 5.8 | 6.618 | 6.7 | 7.6 | 10.0 | 1.306 |
| screenTime | 1.9 | 4.0 | 5.144 | 5.0 | 6.0 | 10.0 | 1.809 |
| workHours | 4.5 | 4.5 | 5.802 | 4.5 | 6.0 | 14.0 | 2.447 |
| activity | 0.0 | 2.8 | 9.543 | 7.9 | 13.2 | 180.0 | 10.548 |
| social | 1.0 | 3.6 | 4.887 | 4.7 | 6.2 | 10.0 | 1.99 |

## Wellbeing and Domain Scores

- Overall wellbeing mean: 68.966
- Overall wellbeing min/max: 37 / 91

| Domain | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| physical | 61.908 | 15 | 96 |
| mental | 65.336 | 12 | 100 |
| social | 48.875 | 10 | 100 |
| occupational | 82.628 | 27 | 100 |
| digital | 87.823 | 41 | 100 |

## Feature Correlation With Wellbeing Score

| Feature | Correlation |
| --- | ---: |
| sleep | 0.255 |
| stress | -0.738 |
| mood | 0.408 |
| energy | 0.826 |
| screenTime | -0.065 |
| workHours | -0.156 |
| activity | 0.172 |
| social | 0.305 |

## Notes

- This EDA uses the processed app-ready CSV, not the raw StudentLife files.
- The current score is a research prototype score, not a clinical measure.
- Correlations are exploratory and should not be interpreted as causal.
