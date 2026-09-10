# VarunaPath AI — Prototype Datasets Dictionary

All datasets contained in this directory are simulated demonstration datasets created specifically for the VarunaPath AI maritime decision support prototype. 

> **Important Classification Notice:**  
> All records are classified as **Prototype Data**. They do not represent audited operational metrics, live vessel AIS telemetry, confirmed commercial trade contracts, or real-world customs filings.

## Datasets Directory

| File Name | Description | Key Columns |
|---|---|---|
| `ports_master.csv` | Indian discharge ports master data for Import workflow | Port, Handling ₹/t, Waiting Days, Max Vessel DWT, Port Risk, Coordinates, Berths |
| `sample_cargo_demand.csv` | Historical monthly commodity demand for Import forecasting | Month, Demand (000 t), Commodity, Origin, Status |
| `export_orders.csv` | Export buyer orders with quantities, inventory, and laycan | Order ID, Buyer Reference, Buyer Country, Commodity, Order Quantity, Incoterm |
| `export_inventory.csv` | Stockyard inventory and domestic reservation reserves | Commodity, Location, Current Stock, Quality Grade, Reserved Domestic |
| `production_schedule.csv` | Scheduled plant production preceding vessel loading | Schedule ID, Commodity, Daily Output, Target Completion, Committed Quantity |
| `export_loading_ports.csv` | Indian East Coast loading ports specifications | Port Name, Max Draft, Max DWT, Handling Rate, Loading Capacity, Port Risk |
| `foreign_destination_ports.csv` | Overseas destination discharge ports specifications | Port Name, Country, Max Draft, Discharge Rate, Port Tariff, Distance, Risk |
| `export_routes.csv` | Nautical distances, sailing days, and chokepoints | Route ID, Loading Port, Destination Port, Distance (nm), Sailing Days, Risk |
| `export_vessels.csv` | Bulk carrier fleet characteristics for export fixtures | Vessel Name, Class, DWT Capacity, Draft, Charter Cost, Speed, Fuel Cons |
| `export_freight_rates.csv` | Benchmark export freight rates and volatility metrics | Commodity, Destination Market, Benchmark Freight, Volatility Index |
| `export_costs.csv` | Incoterm-specific cost components and tariff benchmarks | Cost Component, Inland Rate, Port Rate, Documentation, Insurance, Demurrage |
| `export_documents.csv` | Export documentation checklist and compliance statuses | Document ID, Document Name, Mandatory Status, Prototype Default Status |
