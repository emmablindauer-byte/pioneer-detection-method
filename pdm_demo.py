"""
Pioneer Detection Method (PDM) - examples
==============================

A minimal Python implementation of the Pioneer Detection Method (PDM)—
a convergence-based expert-aggregation algorithm designed for environments
with structural change and heterogeneous learning speeds. The method
identifies “pioneers”: experts whose predictions or opinions deviate early
but toward whom others subsequently converge.

This simplified version is intentionally lightweight and pedagogical:

- no input validation
- no missing-value handling
- fixed cross-sectional benchmark: mean of all other experts
- orientation measured through single-period slopes (first differences)
- weights rescaled to sum to 1 whenever any pioneer is detected

The algorithm follows the three canonical PDM steps:

Step 1 — Distance condition
    An expert moves closer to the group:
    |x_i^t − m_-i^t| < |x_i^{t−1} − m_-i^{t−1}|

Step 2 — Orientation condition
    The group moves more toward the expert than the expert moves toward the group:
    |Δm_-i^t| > |Δx_i^t|

Step 3 — Proportion condition
    Relative contribution of the group’s movement:
    |Δm_-i^t| / (|Δm_-i^t| + |Δx_i^t|)

Weights are computed from Step 3 and normalized across experts at each t.
If no pioneer exists at time t, the pooled estimate defaults to the simple
cross-sectional mean.

This code corresponds to the approach introduced in:
    Vansteenberghe, Eric (2025),
    "Insurance Supervision under Climate Change: A Pioneer Detection Method,"
    The Geneva Papers on Risk and Insurance – Issues and Practice,
    https://doi.org/10.1057/s41288-025-00367-y

The PDM is applicable beyond insurance supervision—including adaptive
forecasting under non-stationarity and multi-agent systems where early
detectors of a regime shift must guide collective behavior
(e.g., drone swarms, robotic fleets, autonomous-vehicle coordination).
"""


from pdm import compute_pioneer_weights_simple, pooled_forecast_simple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO


# --------------------------------------------------------------
# Simple example with 3 experts and a clear pioneer
# Expert E1 is the pioneer: it moves early and smoothly upward.
# Experts E2 and E3 start far below and then converge quickly
# towards E1 from t=2 onwards.
# --------------------------------------------------------------

data = {
    "E1_pioneer":  [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],  # early, steady trend
    "E2_follower": [0.5, 0.5, 0.9, 1.2, 1.3, 1.4],  # jumps up toward E1
    "E3_follower": [0.4, 0.4, 0.8, 1.1, 1.2, 1.3],  # similar convergence
}
idx = pd.RangeIndex(start=0, stop=6)  # time: 0,...,5
forecasts_example = pd.DataFrame(data, index=idx)

weights_example = compute_pioneer_weights_simple(forecasts_example)
pooled_example = pooled_forecast_simple(forecasts_example, weights_example)

print("Forecasts:")
print(forecasts_example, "\n")

print("Pioneer weights:")
print(weights_example, "\n")

print("Pooled forecast (Pioneer Detection Method, with mean fallback):")
print(pooled_example)



"""
Example time series with a clear pioneer expert.

We construct a three-expert panel (length 100) as follows.

1. Baseline path
   Let t = 0, …, 99 and define a baseline b_t = 0.1 * t.

2. Expert 1 (pioneer)
   - For t <= 30, expert 1 follows the baseline: x_1(t) = 0.1 * t.
   - For 30 < t <= 60, expert 1 anticipates a regime shift and moves onto
     a steeper linear path:
         x_1(t) = 0.1 * 30 + 0.4 * (t - 30).
   - For t > 60, expert 1 plateaus at the value reached at t = 60:
         x_1(t) = 0.1 * 30 + 0.4 * (60 - 30).

3. Experts 2 and 3 (followers)
   - Initially they follow the same baseline b_t.
   - From t = 45 onward, they gradually converge towards expert 1 using
     a simple adjustment rule
         x_j(t+1) = x_j(t) + α_j * (x_1(t) - x_j(t)),  j ∈ {2,3}
     with α_2 = 0.12 and α_3 = 0.06.

This creates a phase where |x_j(t) - x_1(t)| shrinks over time and the slopes
of experts 2 and 3 are oriented towards expert 1, so the distance-reduction
and orientation conditions of the Pioneer Detection Method are naturally
satisfied.
"""


# --- Load the explicit 3-expert time series and apply PDM --------------------

csv_data = """t,exp1,exp2,exp3
0,0.0,0.0,0.0
1,0.1,0.1,0.1
2,0.2,0.2,0.2
3,0.3,0.3,0.3
4,0.4,0.4,0.4
5,0.5,0.5,0.5
6,0.6,0.6,0.6
7,0.7,0.7,0.7
8,0.8,0.8,0.8
9,0.9,0.9,0.9
10,1.0,1.0,1.0
11,1.1,1.1,1.1
12,1.2,1.2,1.2
13,1.3,1.3,1.3
14,1.4,1.4,1.4
15,1.5,1.5,1.5
16,1.6,1.6,1.6
17,1.7,1.7,1.7
18,1.8,1.8,1.8
19,1.9,1.9,1.9
20,2.0,2.0,2.0
21,2.1,2.1,2.1
22,2.2,2.2,2.2
23,2.3,2.3,2.3
24,2.4,2.4,2.4
25,2.5,2.5,2.5
26,2.6,2.6,2.6
27,2.7,2.7,2.7
28,2.8,2.8,2.8
29,2.9,2.9,2.9
30,3.0,3.0,3.0
31,3.4,3.1,3.1
32,3.8,3.2,3.2
33,4.2,3.3,3.3
34,4.6,3.4,3.4
35,5.0,3.5,3.5
36,5.4,3.6,3.6
37,5.8,3.7,3.7
38,6.2,3.8,3.8
39,6.6,3.9,3.9
40,7.0,4.0,4.0
41,7.4,4.1,4.1
42,7.8,4.2,4.2
43,8.2,4.3,4.3
44,8.6,4.4,4.4
45,9.0,4.5,4.5
46,9.4,4.599,4.527
47,9.8,4.687,4.557
48,10.2,4.766,4.589
49,10.6,4.837,4.623
50,11.0,4.901,4.659
51,11.4,4.958,4.698
52,11.8,5.01,4.738
53,12.2,5.057,4.781
54,12.6,5.1,4.826
55,13.0,5.14,4.873
56,13.4,5.177,4.923
57,13.8,5.211,4.975
58,14.2,5.242,5.029
59,14.6,5.271,5.086
60,15.0,5.298,5.145
61,15.0,5.322,5.206
62,15.0,5.345,5.27
63,15.0,5.366,5.336
64,15.0,5.386,5.404
65,15.0,5.404,5.474
66,15.0,5.422,5.547
67,15.0,5.438,5.622
68,15.0,5.453,5.699
69,15.0,5.467,5.779
70,15.0,5.48,5.861
71,15.0,5.493,5.945
72,15.0,5.504,6.032
73,15.0,5.514,6.121
74,15.0,5.524,6.214
75,15.0,5.533,6.309
76,15.0,5.541,6.407
77,15.0,5.548,6.507
78,15.0,5.554,6.611
79,15.0,5.559,6.717
80,15.0,5.564,6.826
81,15.0,5.568,6.939
82,15.0,5.571,7.054
83,15.0,5.574,7.173
84,15.0,5.577,7.295
85,15.0,5.579,7.42
86,15.0,5.581,7.549
87,15.0,5.583,7.681
88,15.0,5.584,7.817
89,15.0,5.586,7.956
90,15.0,5.587,8.099
91,15.0,5.588,8.245
92,15.0,5.589,8.395
93,15.0,5.59,8.549
94,15.0,5.59,8.707
95,15.0,5.591,8.869
96,15.0,5.591,9.035
97,15.0,5.592,9.204
98,15.0,5.592,9.378
99,15.0,5.593,9.555
"""

df = pd.read_csv(StringIO(csv_data))
t = df["t"]
forecasts_ts = df[["exp1", "exp2", "exp3"]]

weights_ts = compute_pioneer_weights_simple(forecasts_ts)
pdm_ts = pooled_forecast_simple(forecasts_ts, weights_ts)
mean_ts = forecasts_ts.mean(axis=1)

# --- Plot the three expert series, simple mean, and PDM estimate -------------

plt.figure(figsize=(8, 4))

for col in forecasts_ts.columns:
    plt.plot(t, forecasts_ts[col], label=col)

plt.plot(t, mean_ts, linestyle=":", linewidth=2, label="Simple mean")
plt.plot(t, pdm_ts, linestyle="--", linewidth=2, label="PDM pooled forecast")

plt.xlabel("Time")
plt.ylabel("Forecast")
plt.title("Three-expert panel: experts, simple mean, and PDM pooled forecast")
plt.legend()
plt.tight_layout()
plt.show()
