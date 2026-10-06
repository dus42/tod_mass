# Aircraft mass at top of descent from ADS-B

Estimates the aircraft mass at the top of descent (TOD) from descent ADS-B trajectories, using only openly available inputs: the trajectory, the aircraft type from the registry, and published aircraft properties (MTOW and OEW from OpenAP). It accompanies the paper *Estimating Aircraft Mass at Top of Descent from ADS-B Trajectory Data* (14th OpenSky Symposium, 2026).

The models were trained on 464,633 flights of the [EUROCONTROL PRC Data Challenge 2024](https://ansperformance.eu/study/data-challenge/), with reference TOD masses obtained by propagating the disclosed takeoff weight with the [OpenAP](https://openap.dev) fuel flow model. The ADS-B-only model reaches a MAPE of 3.39 %; with weather and flight information, 2.55 %.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)
- **An OpenSky Network account with Trino access**, to download ADS-B data (see below)
- The trained models, from Zenodo: **[DOI to be added]**, placed in `models/`

```bash
git clone https://github.com/dus42/tod_mass.git
cd tod_mass
uv sync
```

## OpenSky Trino access

Historical ADS-B data are queried from the [OpenSky Network](https://opensky-network.org) Trino database, which is available to university-affiliated researchers, governmental organisations and aviation authorities.

See the [Trino documentation](https://openskynetwork.github.io/opensky-api/trino.html) and the [pyopensky credentials guide](https://mode-s.org/pyopensky/credentials.html). On first use, a browser login may be requested.

## Usage

Download a period of arrivals and estimate the TOD mass of each flight:

```bash
uv run python estimate_tod_from_opensky.py --arrival-airport EHAM \
    --start "2026-01-01 00:00" --stop "2026-01-01 11:59:59" --bounds 1,48,8,56
```

| Option | Meaning |
|---|---|
| `--arrival-airport` | ICAO code of the arrival airport |
| `--start`, `--stop` | UTC time window |
| `--bounds` | `west,south,east,north` box in degrees; limits the download to the part of each flight near the airport |
| `--weather` | interpolate ERA5 winds with [fastmeteo](https://github.com/open-aviation/fastmeteo); more accurate, but downloads and caches the weather grid |
| `--input`, `--flight-list` | use files already downloaded instead of querying OpenSky |

The output CSV has one row per flight: `flight_id, icao24, callsign, aircraft_type, wtc, adep, ades, tod_time, tod_lat, tod_lon, tod_alt, tod_mass_estimate, unseen_type, unseen_airport`. The last two columns flag aircraft types or airports the model was not trained on; those estimates are the least reliable.

For your own trajectory files, `pipeline_tod.py` runs the same feature extraction and prediction:

```bash
uv run python pipeline_tod.py --input trajectories.parquet \
    --model model_tod_noweather_all_nomassrange.txt
```

## Models

| Model | Inputs | Test MAPE |
|---|---|---|
| `model_tod_noweather_all_nomassrange` | ADS-B only | 3.39 % |
| `model_tod_weather_all_nomassrange` | ADS-B and weather | 3.15 % |
| `model_tod_noweather_all_finfo_nomassrange` | ADS-B and flight information | 2.72 % |
| `model_tod_weather_all_finfo_nomassrange` | ADS-B, weather and flight information | 2.55 % |

`estimate_tod_from_opensky.py` picks the model that matches the available inputs. The models predict the TOD mass as a fraction of the MTOW, so they also apply to aircraft types absent from training, with a larger error.

## Reproducing the paper

The `paper/` folder contains the LaTeX source and the code that produces every figure and table; see [`paper/README.md`](paper/README.md).

## Built on

- [OpenSky Network](https://opensky-network.org) and its [Trino database](https://openskynetwork.github.io/opensky-api/trino.html)
- [pyopensky](https://mode-s.org/pyopensky/) and [traffic](https://traffic-viz.github.io) for querying and processing ADS-B data
- [OpenAP](https://openap.dev) for aircraft properties, fuel flow and flight phases
- [fastmeteo](https://github.com/open-aviation/fastmeteo) for ERA5 weather along trajectories
- [LightGBM](https://lightgbm.readthedocs.io) for the regression models

## Citation

[To be added after publication.]

## License

MIT, see [LICENSE](LICENSE).
