"""Statistics and figures of the Ford Dry Lake case study (shared by the notebook and the PDF report).

Reads the small files in results/ (InSAR4SM and ERA5-Land, see prepare_results.py) and the in-situ
file that the user downloaded from https://ismn.earth. ISMN data are not part of this repository.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from insar4sm.ismn import read_ismn

DRY_STATE = 2.8          # % vol., SM_point.sm_dry_state; dry dates are capped at DRY_STATE + 1
WET_THRESHOLD = 6.0      # % vol. in situ, splits dry and wet dates (chosen after the run)
FIGURES_DIR = os.path.join(HERE, 'figures')

STATION = {'name': 'Ford Dry Lake (SCAN)', 'lon': -115.09763, 'lat': 33.65470}

# ---------------------------------------------------------------- style
# Validated categorical slots (light surface): blue / orange / aqua. Reference data in neutral ink.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_3 = '#8a8984'
GRID = '#e6e5e1'
C_INSAR = '#2a78d6'      # InSAR4SM (Sentinel-1)
C_ERA5 = '#eb6834'       # ERA5-Land
C_NISAR = '#1baf7a'      # InSAR4SM (NISAR); secondary encoding: square marker
C_INSITU = '#3d3c39'     # ISMN in situ (reference)
SENSOR_STYLE = {'S1': dict(color=C_INSAR, marker='o', label='InSAR4SM Sentinel-1 (C-band)'),
                'NISAR': dict(color=C_NISAR, marker='s', label='InSAR4SM NISAR (L-band)')}
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def set_style():
    matplotlib.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.labelsize': 9, 'xtick.color': INK_2, 'ytick.color': INK_2,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 8, 'legend.labelcolor': INK_2,
        'font.family': 'DejaVu Sans', 'lines.linewidth': 2, 'figure.dpi': 110,
    })


# ---------------------------------------------------------------- data
def load_results(ismn_file, results_dir=RESULTS_DIR):
    """Returns a dict with tracks, comparisons, estimates, daily reference and run info.

    ismn_file: 5 cm soil moisture of station FordDryLake (network SCAN), downloaded with your own
    ISMN account (.stm, or CSV with columns date_time, soil_moisture).
    """
    if not ismn_file or not os.path.exists(ismn_file):
        raise FileNotFoundError('ISMN file not found. Download station FordDryLake (network SCAN, 5 cm) from '
                                'https://ismn.earth with your own account and pass its path.')
    ismn = read_ismn(ismn_file) * 100
    tracks = pd.read_csv(os.path.join(results_dir, 'tracks.csv'), dtype={'track': str})
    comp = pd.read_csv(os.path.join(results_dir, 'insar4sm_era5.csv'), parse_dates=['time_utc'], dtype={'track': str})
    # in-situ value at (or last before) the acquisition hour
    comp['insitu_5cm'] = ismn.asof(pd.DatetimeIndex(comp['time_utc'])).values
    comp['date'] = comp['time_utc'].dt.normalize()
    est = pd.read_csv(os.path.join(results_dir, 'estimates.csv.gz'), parse_dates=['date'], dtype={'track': str})
    daily = pd.read_csv(os.path.join(results_dir, 'era5_daily.csv'), parse_dates=['date'], index_col='date')
    daily['ismn_5cm_pct'] = ismn.resample('D').mean().reindex(daily.index)
    with open(os.path.join(results_dir, 'run_info.json')) as f:
        run_info = json.load(f)
    # tracks with an estimate (a track whose frame barely touches the AOI has none)
    valid = comp.groupby(['sensor', 'track'])['insar4sm_aoi_mean'].apply(lambda s: s.notna().any())
    valid_tracks = valid[valid].index.tolist()
    comp['track_label'] = comp['sensor'] + ' ' + comp['track']
    tracks['track_label'] = tracks['sensor'] + ' ' + tracks['track']
    tracks['used'] = [(s, t) in valid_tracks for s, t in zip(tracks.sensor, tracks.track)]
    return {'tracks': tracks, 'comp': comp, 'est': est, 'daily': daily, 'run_info': run_info,
            'valid_tracks': valid_tracks}


def scores(obs, est):
    """Bias, RMSE, unbiased RMSE and Pearson R of est vs obs (both in %)."""
    ok = pd.concat([obs, est], axis=1).dropna()
    if len(ok) < 3:
        return dict(N=len(ok), bias=np.nan, RMSE=np.nan, ubRMSE=np.nan, R=np.nan)
    o, e = ok.iloc[:, 0].values, ok.iloc[:, 1].values
    d = e - o
    return dict(N=len(ok), bias=d.mean(), RMSE=np.sqrt(np.mean(d**2)),
                ubRMSE=np.sqrt(np.mean((d - d.mean())**2)), R=np.corrcoef(o, e)[0, 1])


def metrics_table(data):
    """Scores per track and pooled over all tracks, for InSAR4SM (AOI mean, station cell) and ERA5-Land."""
    comp = data['comp'][data['comp'].set_index(['sensor', 'track']).index.isin(data['valid_tracks'])]
    groups = [(label, g) for label, g in comp.groupby('track_label', sort=False)]
    groups += [('All Sentinel-1', comp[comp.sensor == 'S1']), ('All NISAR', comp[comp.sensor == 'NISAR']),
               ('All tracks', comp)]
    rows = []
    for label, g in groups:
        for product, col in [('InSAR4SM (AOI mean)', 'insar4sm_aoi_mean'),
                             ('InSAR4SM (station cell)', 'insar4sm_station_cell'),
                             ('ERA5-Land', 'era5_swvl1')]:
            rows.append({'tracks': label, 'product': product, **scores(g['insitu_5cm'], g[col])})
    return pd.DataFrame(rows)


def wetness_table(data):
    """Scores on dry and wet dates (in situ below / above WET_THRESHOLD), all tracks pooled."""
    comp = data['comp'].dropna(subset=['insar4sm_aoi_mean'])
    rows = []
    for label, g in [('in situ < {:.0f} %'.format(WET_THRESHOLD), comp[comp.insitu_5cm < WET_THRESHOLD]),
                     ('in situ >= {:.0f} %'.format(WET_THRESHOLD), comp[comp.insitu_5cm >= WET_THRESHOLD]), ('all dates', comp)]:
        for product, col in [('InSAR4SM (AOI mean)', 'insar4sm_aoi_mean'), ('ERA5-Land', 'era5_swvl1')]:
            rows.append({'dates': label, 'product': product, **scores(g['insitu_5cm'], g[col])})
    return pd.DataFrame(rows)


def dry_state_share(data):
    """Share of the AOI means close to the prescribed dry state (DRY_STATE -0.3 to +1 %), per sensor."""
    comp = data['comp'].dropna(subset=['insar4sm_aoi_mean'])
    near = comp['insar4sm_aoi_mean'].between(DRY_STATE - 0.3, DRY_STATE + 1)
    return near.groupby(comp['sensor']).mean()


def headline(data):
    """Key numbers for the summary."""
    m = metrics_table(data)
    allt = m[m.tracks == 'All tracks'].set_index('product')
    comp = data['comp'][data['comp'].set_index(['sensor', 'track']).index.isin(data['valid_tracks'])]
    dates = np.sort(comp['date'].dt.normalize().unique())
    gaps = np.diff(dates).astype('timedelta64[D]').astype(int)
    return {'rmse_insar': allt.loc['InSAR4SM (AOI mean)', 'RMSE'],
            'r_insar': allt.loc['InSAR4SM (AOI mean)', 'R'],
            'rmse_era5': allt.loc['ERA5-Land', 'RMSE'],
            'r_era5': allt.loc['ERA5-Land', 'R'],
            'ubrmse_insar': allt.loc['InSAR4SM (AOI mean)', 'ubRMSE'],
            'ubrmse_era5': allt.loc['ERA5-Land', 'ubRMSE'],
            'n_obs': int(allt.loc['InSAR4SM (AOI mean)', 'N']),
            'n_tracks': len(data['valid_tracks']),
            'n_dates': len(dates), 'median_gap_days': float(np.median(gaps)),
            'max_gap_days': int(gaps.max()),
            'n_cells': int(data['est'].groupby(['sensor', 'track'])['cell'].nunique().max())}


def local_km(lon, lat):
    """East / north distance from the station in km (UTM 11N)."""
    t = Transformer.from_crs(4326, 32611, always_xy=True)
    x0, y0 = t.transform(STATION['lon'], STATION['lat'])
    x, y = t.transform(np.asarray(lon), np.asarray(lat))
    return (x - x0) / 1000, (y - y0) / 1000


# ---------------------------------------------------------------- figures
def _save(fig, name, save):
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, name + '.png'), dpi=200, bbox_inches='tight')
    return fig


def fig_overview_timeseries(data, save=True):
    """Hero figure: rain, in-situ soil moisture, ERA5-Land and every InSAR4SM estimate (all tracks)."""
    set_style()
    daily, comp = data['daily'], data['comp']
    fig, (ax_p, ax) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={'height_ratios': [1, 3.2], 'hspace': 0.08})
    ax_p.bar(daily.index, daily['era5_precip_mm'], width=1.0, color=INK_3, linewidth=0)
    ax_p.set_ylabel('Rain\n(mm/day)')
    ax_p.set_title('Soil moisture at Ford Dry Lake, Nov 2025 – Nov 2026', pad=8)

    ax.plot(daily.index, daily['ismn_5cm_pct'], color=C_INSITU, lw=1.6, label='In situ 5 cm (ISMN, daily mean)', zorder=3)
    ax.plot(daily.index, daily['era5_swvl1_pct'], color=C_ERA5, lw=1.4, label='ERA5-Land layer 1 (0–7 cm, 9 km)', zorder=2)
    for sensor in ['S1', 'NISAR']:
        g = comp[comp.sensor == sensor].dropna(subset=['insar4sm_aoi_mean'])
        st = SENSOR_STYLE[sensor]
        ax.scatter(g['date'], g['insar4sm_aoi_mean'], s=22, color=st['color'], marker=st['marker'],
                   edgecolor=SURFACE, linewidth=0.8, label=st['label'] + ', AOI mean', zorder=4)
    ax.set_ylabel('Volumetric soil moisture (%)')
    ax.set_ylim(0, float(np.nanmax(daily[['ismn_5cm_pct', 'era5_swvl1_pct']].values)) + 2)
    ax.legend(loc='upper right', ncol=1)
    fig.autofmt_xdate(rotation=0, ha='center')
    return _save(fig, 'fig1_overview_timeseries', save)


def fig_tracks_timeseries(data, save=True):
    """Small multiples, one panel per track: InSAR4SM AOI mean and ERA5-Land vs in situ."""
    set_style()
    daily, comp = data['daily'], data['comp']
    labels = [f'{s} {t}' for s, t in data['valid_tracks']]
    m = metrics_table(data).set_index(['tracks', 'product'])
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 1.9 * len(labels)), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.45})
    for ax, label in zip(np.atleast_1d(axes), labels):
        g = comp[comp.track_label == label]
        st = SENSOR_STYLE[label.split()[0]]
        ax.plot(daily.index, daily['ismn_5cm_pct'], color=C_INSITU, lw=1.3, label='In situ 5 cm')
        ax.plot(g['date'], g['era5_swvl1'], color=C_ERA5, lw=0, marker='D', ms=3.5, label='ERA5-Land')
        ax.plot(g['date'], g['insar4sm_aoi_mean'], color=st['color'], lw=1.2, marker=st['marker'], ms=4.5,
                mec=SURFACE, mew=0.6, label='InSAR4SM (AOI mean)')
        s_i, s_e = m.loc[(label, 'InSAR4SM (AOI mean)')], m.loc[(label, 'ERA5-Land')]
        ax.set_title('{}  ·  {} dates  ·  InSAR4SM RMSE {:.1f} %, R {:.2f}  |  ERA5-Land RMSE {:.1f} %, R {:.2f}'.format(
            label, int(s_i.N), s_i.RMSE, s_i.R, s_e.RMSE, s_e.R), fontsize=9, pad=4)
        ax.set_ylim(0, 35)
    np.atleast_1d(axes)[0].legend(loc='upper right', ncol=3)
    fig.supylabel('Volumetric soil moisture (%)', fontsize=9, color=INK_2, x=0.07)
    return _save(fig, 'fig2_tracks_timeseries', save)


def fig_scatter(data, save=True):
    """InSAR4SM and ERA5-Land against in situ, all tracks, with 1:1 line."""
    set_style()
    comp = data['comp'][data['comp'].set_index(['sensor', 'track']).index.isin(data['valid_tracks'])]
    m = metrics_table(data).set_index(['tracks', 'product'])
    lim = (0, max(28, float(np.nanmax(comp[['insitu_5cm', 'insar4sm_aoi_mean', 'era5_swvl1']].values)) + 2))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharex=True, sharey=True)
    for ax, (title, col, key) in zip(axes, [('InSAR4SM (AOI mean)', 'insar4sm_aoi_mean', 'InSAR4SM (AOI mean)'),
                                             ('ERA5-Land (0–7 cm, 9 km)', 'era5_swvl1', 'ERA5-Land')]):
        ax.plot(lim, lim, color=INK_3, lw=1, ls='--', zorder=1)
        if col == 'era5_swvl1':
            ax.scatter(comp['insitu_5cm'], comp[col], s=18, color=C_ERA5, marker='D', edgecolor=SURFACE, linewidth=0.6, zorder=3)
        else:
            for sensor in ['S1', 'NISAR']:
                g = comp[comp.sensor == sensor]
                st = SENSOR_STYLE[sensor]
                ax.scatter(g['insitu_5cm'], g[col], s=20, color=st['color'], marker=st['marker'],
                           edgecolor=SURFACE, linewidth=0.6, label=st['label'], zorder=3)
            ax.legend(loc='upper left')
        s = m.loc[('All tracks', key)]
        ax.text(0.97, 0.04, 'N = {}\nRMSE = {:.1f} %\nbias = {:+.1f} %\nR = {:.2f}'.format(int(s.N), s.RMSE, s.bias, s.R),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=INK)
        ax.set_title(title)
        ax.set_xlabel('In situ 5 cm (%)')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    axes[0].set_ylabel('Estimated (%)')
    return _save(fig, 'fig3_scatter', save)


def fig_rmse_bars(data, save=True):
    """RMSE per track: InSAR4SM (AOI mean) vs ERA5-Land."""
    set_style()
    m = metrics_table(data)
    order = [f'{s} {t}' for s, t in data['valid_tracks']] + ['All tracks']
    piv = m.pivot(index='tracks', columns='product', values='RMSE').loc[order]
    y = np.arange(len(order))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for off, prod, color in [(-h / 2 - 0.01, 'InSAR4SM (AOI mean)', C_INSAR), (h / 2 + 0.01, 'ERA5-Land', C_ERA5)]:
        bars = ax.barh(y + off, piv[prod], height=h, color=color, label=prod)
        for b, v in zip(bars, piv[prod]):
            ax.text(v + 0.06, b.get_y() + b.get_height() / 2, '{:.1f}'.format(v), va='center', fontsize=8, color=INK_2)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel('RMSE against in situ 5 cm (% vol.)  — lower is better')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, float(piv.values.max()) * 1.12)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.3)
    ax.set_title('Error per track', pad=26)
    return _save(fig, 'fig4_rmse_bars', save)


def fig_maps(data, sensor='S1', track=None, save=True):
    """Soil moisture maps over the AOI on the driest and the wettest date of one track."""
    set_style()
    est = data['est']
    if track is None:  # the track with most dates
        track = est[est.sensor == sensor].groupby('track')['date'].nunique().idxmax()
    g = est[(est.sensor == sensor) & (est.track == track)]
    mean_by_date = g.groupby('date')['sm_pct'].mean()
    dates = [mean_by_date.idxmin(), mean_by_date.idxmax()]
    vmax = float(np.nanpercentile(g[g.date.isin(dates)]['sm_pct'], 98))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
    for ax, date, title in zip(axes, dates, ['Driest date', 'Wettest date']):
        d = g[g.date == date]
        x, y = local_km(d['lon'], d['lat'])
        # estimates sit on a regular 125 m lattice: index them from the lattice origin, one filled cell each
        ix, iy = np.round((x - x.min()) / 0.125), np.round((y - y.min()) / 0.125)
        grid = pd.DataFrame({'x': x.min() + ix * 0.125, 'y': y.min() + iy * 0.125, 'v': d['sm_pct'].values})
        piv = grid.pivot_table(index='y', columns='x', values='v')
        sc = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap=SEQ_BLUE, vmin=0, vmax=vmax, shading='nearest')
        ax.scatter([0], [0], marker='*', s=170, color=SURFACE, edgecolor=INK, linewidth=1.0, zorder=5)
        ax.annotate('ISMN station', (0, 0), xytext=(7, 7), textcoords='offset points', fontsize=8, color=INK,
                    bbox=dict(boxstyle='round,pad=0.2', fc=SURFACE, ec='none', alpha=0.85))
        ax.set_title('{}: {}  (AOI mean {:.1f} %)'.format(title, date.strftime('%Y-%m-%d'), mean_by_date[date]), fontsize=10)
        ax.set_xlabel('East of station (km)')
        ax.set_aspect('equal')
        ax.grid(False)
    axes[0].set_ylabel('North of station (km)')
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label('Volumetric soil moisture (%)', fontsize=9, color=INK_2)
    cb.outline.set_visible(False)
    fig.suptitle('InSAR4SM soil moisture maps, {} track {} (an estimate every 125 m, each from a 250 m window)'.format(sensor, track),
                 x=0.02, ha='left', fontsize=11, fontweight='bold', color=INK)
    return _save(fig, 'fig5_maps', save)


def all_figures(data, save=True):
    return [fig_overview_timeseries(data, save), fig_tracks_timeseries(data, save), fig_scatter(data, save),
            fig_rmse_bars(data, save), fig_maps(data, save=save)]
