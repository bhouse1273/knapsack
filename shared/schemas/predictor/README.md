# Predictor schemas

* `item_scores.schema.json` – Arrow schema for batch scoring output that feeds the solver.
* `score.proto` – gRPC contract for online scoring.

Whenever these files change, bump `version` inside the JSON schema and update
`PredictorSchema` fixtures plus knapsack consumers to reject incompatible payloads.
