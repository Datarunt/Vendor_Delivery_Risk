Vendor Delivery Risk Predictor · system architecture

# Vendor Commit Risk Architecture

How three uploaded spreadsheets become a HIGH/LOW risk file. The app runs on one computer as a small web app; each button press runs one stage, and the results of stage 1 wait in the server's memory until stage 2.

Blue marks the path that decides the label: the miss model's chance of being met goes to `risk_engine.py`, which flags anything under 85% as HIGH. The quantity forecast from stage 1 no longer sets the label, but it still controls which commits appear, shown in red.

## Two buttons, two stages

“Predict Risk” cleans the history and builds the quantity forecast. “Combine” trains the miss model and labels the commits. Stage 1 results sit in the server's memory in between, which is why only one person can use a running copy.

## Where the label comes from

History rows travel from stage 1 into stage 2. The scorecard and the miss model are both built from them on every run; nothing is saved between runs.

## Scoring happens separately

`performance_check.py` is a separate script run weeks later. It compares the Risk column with actual receipts and produces the accuracy figures.