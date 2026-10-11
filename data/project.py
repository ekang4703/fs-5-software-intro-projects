#step 1
import polars as pl
#step 2
import matplotlib.pyplot as plt
import numpy as np

#step 1
df = pl.read_parquet('software-data.parquet')

df_filled = df.select(pl.all().forward_fill())

with pl.Config(tbl_rows=-1):
    print(df_filled.head(50))
    #print(df_filled)


#step 2
calc_speed = df_filled["SME_TRQSPD_Speed"] * (12/41) * ((2*np.pi)/60) * 0.2
plt.plot(df_filled["Time"], calc_speed)
plt.title("Speed VS Time")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.grid(True)
plt.show()
#df_filled.row()
filtered_df = df_filled.filter(
    (pl.col("Time") >= 10)
)
indexpos_firstrow = (df_filled["Time"] >= 10).arg_true()[0]
speed_at_ten = calc_speed.item(indexpos_firstrow)
print(f"Car's speed at 10s is {speed_at_ten}")

#step 3
#accelration
speed_change = calc_speed.diff(n=10)
deriv = df_filled["Time"].diff(n=10)
acceleration = speed_change / deriv
accelerating = (acceleration > 0).fill_null(False)

#braking
braking = df_filled["ETC_STATUS_BRAKE_SENSE_VOLTAGE"] >= 375

#coasting
coasting = (df_filled["SME_THROTL_TorqueDemand"] == 0) & (~braking) & (calc_speed > 0)

def get_state_intervals(df, state_series):
    df_temp = df.with_columns([
        state_series.alias("is_active"),
        (state_series != state_series.shift(1)).fill_null(True).cum_sum().alias("block_id")
    ])
    
    return (
        df_temp.filter(pl.col("is_active"))
        .group_by("block_id", maintain_order=True)
        .agg([
            pl.col("Time").min().alias("start_s"),
            pl.col("Time").max().alias("end_s"),
            (pl.col("Time").max() - pl.col("Time").min()).alias("duration_s")
        ])
    )

acc_intervals = get_state_intervals(df_filled, accelerating)
brake_intervals = get_state_intervals(df_filled, braking)
coast_intervals = get_state_intervals(df_filled, coasting)

print("acceleration intervals")
print(acc_intervals)

print("\n Breaking intervals")
print(brake_intervals)

print("\n coasting intervals")
print(coast_intervals)

#acc graph
plt.plot(df_filled["Time"], accelerating, color="green")
plt.title("Accelerating State vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Active (1/0)")
plt.grid(True)
plt.show()

#brake graph
plt.plot(df_filled["Time"], braking, color="red")
plt.title("Braking State vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Active (1/0)")
plt.grid(True)
plt.show()

#coast graph
plt.plot(df_filled["Time"], coasting, color="orange")
plt.title("Coasting State vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Active (1/0)")
plt.grid(True)
plt.show()

#step 4
# crossing point
max_lat = df_filled["VDM_GPS_Latitude"].max()
ref_lon = df_filled.filter(pl.col("VDM_GPS_Latitude") == max_lat)["VDM_GPS_Longitude"][0]

# distance from point for every row
distance = ((df_filled["VDM_GPS_Latitude"] - max_lat) ** 2 + (df_filled["VDM_GPS_Longitude"] - ref_lon) ** 2) ** 0.5

# each pass through the crossing point
crossings = get_state_intervals(df_filled, distance < 0.0001)
starts = crossings["start_s"].to_list()
print("STARTS", starts)

# end of driving = last time the car was moving
drive_end = df_filled.filter(calc_speed > 0)["Time"].max()

# laps: crossing to next crossing, then last crossing to end of driving
laps = []
for i in range(len(starts) -1):
    laps.append((starts[i], starts[i+1]))
n_complete = len(laps)
laps.append((starts[-1], drive_end))

print("Crossings:", starts)
for i, (s, e) in enumerate(laps):
    print(f"Lap {i + 1}: {s:.2f} s to {e:.2f} s ({e - s:.2f} s)")

# plot: GPS position colored by time
sc = plt.scatter(df_filled["VDM_GPS_Longitude"], df_filled["VDM_GPS_Latitude"], c=df_filled["Time"], s=4)
plt.colorbar(sc, label="Time (s)")
plt.scatter(df_filled["VDM_GPS_Longitude"][0], df_filled["VDM_GPS_Latitude"][0], color="blue", label="Start")
plt.scatter(ref_lon, max_lat, color="red", label="Crossing point")
plt.title("Track (GPS) colored by time")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.legend()
plt.show()

#step 5
dt_row = df_filled["Time"].diff().mean()

for i, (lap_start, lap_end) in enumerate(laps):
    label = "complete" if i < n_complete else "not complete"

    in_lap = (df_filled["Time"] >= lap_start) & (df_filled["Time"] < lap_end)

    max_speed = calc_speed.filter(in_lap).max()
    max_accel = acceleration.filter(in_lap).max()
    time_accelerating = (accelerating & in_lap).sum() * dt_row
    time_coasting = (coasting & in_lap).sum() * dt_row

    print(f"Lap {i + 1} ({label}): {lap_start:.2f} s to {lap_end:.2f} s")
    print(f"  Max speed: {max_speed:.2f} m/s")
    print(f"  Max acceleration: {max_accel:.2f} m/s^2")
    print(f"  Time accelerating: {time_accelerating:.2f} s")
    print(f"  Time coasting: {time_coasting:.2f} s")

#step 6
# coasting rows at 5 m/s or faster
coast_clean = coasting & (calc_speed >= 5)

#least 1 second
stretches = get_state_intervals(df_filled, coast_clean).filter(pl.col("duration_s") >= 1)
print(stretches)

keep = df_filled["Time"] < 0   # starts all False
for s, e in zip(stretches["start_s"], stretches["end_s"]):
    keep = keep | ((df_filled["Time"] >= s) & (df_filled["Time"] <= e))

v = calc_speed.filter(keep).to_numpy()
a = acceleration.filter(keep).to_numpy()
print("Clean coast-down rows:", len(v))


#step 7
slope, intercept = np.polyfit(v ** 2, a, 1)

m, m_eff, rho, g = 221.4, 244.08, 1.225, 9.81

half_rho_CdA = -slope * m_eff               # 0.5 * rho * Cd * A
CdA = half_rho_CdA / (0.5 * rho)
Crr = -intercept * m_eff / (m * g)

print(f"slope: {slope:.6f}, intercept: {intercept:.4f}")
print(f"0.5*rho*Cd*A: {half_rho_CdA:.4f}")
print(f"Cd*A: {CdA:.4f} m^2")
print(f"Crr: {Crr:.4f}")

# plot the data and the fit
plt.scatter(v ** 2, a, s=4)
plt.plot(v ** 2, slope * v ** 2 + intercept, color="red")
plt.title("Coast-down: acceleration vs speed squared")
plt.xlabel("v^2 (m^2/s^2)")
plt.ylabel("dv/dt (m/s^2)")
plt.show()