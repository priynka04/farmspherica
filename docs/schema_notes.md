# Schema Notes — Farmspherica Nano PAW

Source: NanoPAW_Datasheet__1_.xlsx → Schema sheet
Last updated: 2 June 2026
Real data received: strawberry_records.xlsx (7 days, Plant P01)

## All 28 column definitions

| # | Column Name       | Type    | Unit       | Valid Min | Valid Max | Example         |
|---|-------------------|---------|------------|-----------|-----------|-----------------|
| 1 | date              | DATE    | YYYY-MM-DD | -         | -         | 2026-05-20      |
| 2 | plant_id          | STRING  | -          | P01       | P10       | P01             |
| 3 | week_number       | INTEGER | -          | 1         | 4         | 1               |
| 4 | day_number        | INTEGER | -          | 1         | 30        | 1               |
| 5 | pH                | FLOAT   | pH units   | 4.0       | 9.0       | 6.5             |
| 6 | TDS               | FLOAT   | ppm        | 0         | 5000      | 850             |
| 7 | EC                | FLOAT   | mS/cm      | 0.0       | 5.0       | 1.8             |
| 8 | water_temp_C      | FLOAT   | °C         | 10        | 35        | 22.5            |
| 9 | water_level_cm    | FLOAT   | cm         | 0         | 100       | 18.0            |
|10 | water_colour      | STRING  | dropdown   | -         | -         | Clear           |
|11 | water_smell       | STRING  | dropdown   | -         | -         | None            |
|12 | air_temp_C        | FLOAT   | °C         | 10        | 45        | 28.0            |
|13 | humidity_pct      | FLOAT   | %          | 0         | 100       | 65.0            |
|14 | light_hours       | FLOAT   | hrs/day    | 0         | 24        | 16.0            |
|15 | light_intensity   | STRING  | dropdown   | -         | -         | High            |
|16 | plant_height_cm   | FLOAT   | cm         | 0         | 300       | 12.5            |
|17 | leaf_count        | INTEGER | count      | 0         | 500       | 8               |
|18 | root_colour       | STRING  | dropdown   | -         | -         | White           |
|19 | root_density      | STRING  | dropdown   | -         | -         | Dense           |
|20 | condition         | STRING  | dropdown   | -         | -         | Healthy         |
|21 | deficiency_type   | STRING  | dropdown   | -         | -         | None            |
|22 | stress_symptoms   | STRING  | free text  | -         | -         | None            |
|23 | photo_front       | STRING  | filename   | -         | -         | 2026-05-20_P01_Healthy_Front.jpg |
|24 | photo_side        | STRING  | filename   | -         | -         | 2026-05-20_P01_Healthy_Side.jpg  |
|25 | photo_root        | STRING  | filename   | -         | -         | 2026-05-20_P01_Healthy_Root.jpg  |
|26 | nutrient_formula  | STRING  | dropdown   | -         | -         | Standard_v1     |
|27 | observer          | STRING  | dropdown   | -         | -         | Ambika          |
|28 | remarks           | STRING  | free text  | -         | -         | Plant healthy   |

## Condition labels and what they mean
- HEALTHY — all parameters normal, no action needed
- MILDLY STRESSED — minor stress visible, check pH/EC, monitor next 24h
- DEFICIENT — nutrient deficiency signs, adjust nutrient solution
- CRITICAL — severe decline, immediate action required, alert Tanya

## Observers
- Ambika — fills the Daily Log every day
- Tanya — plant health expert, reviews conditions
- Livia — data quality checks every 2 days

## Photo naming rule
Format: YYYY-MM-DD_PlantID_Condition_Angle.jpg
Example: 2026-05-20_P01_Healthy_Front.jpg
Three angles per plant per day: Front, Side, Root

## Real data received so far
- strawberry_records.xlsx: 7 days (Day 1–7), Plant P01
- pH range observed: 5.8–6.1 (within ideal range)
- EC range observed: 1.0–1.2 mS/cm
- Height growth: 12.5cm → 14.6cm in 7 days
- Status: All healthy, roots white/cream, water clear