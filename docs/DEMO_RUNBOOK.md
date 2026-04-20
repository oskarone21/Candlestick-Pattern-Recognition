# Demo Runbook

This repo is set up so you do not need to retrain before a demo.

## Recommended flow

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo_dashboard.ps1 -RunName demo_balanced_v2_20260420 -Port 3000
```

Then open:

`http://localhost:3000`

## What the script does

- rebuilds the dashboard data snapshot for the saved run
- rebuilds the production dashboard bundle if needed
- pins the dashboard to the chosen run with `DASHBOARD_RUN_NAME`
- starts the production server

## If you want a different saved run

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo_dashboard.ps1 -RunName YOUR_RUN_NAME -Port 3000 -Rebuild
```
