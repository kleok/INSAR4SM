#!/usr/bin/env python3
"""Builds the PDF report of the Ford Dry Lake case study (FordDryLake_case_study.pdf).

Needs reportlab (not part of the InSAR4SM environments): pip install reportlab
    python case_studies/FordDryLake/build_report.py
"""
import os
import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import case_study as cs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'FordDryLake_case_study.pdf')

INK, INK_2, INK_3 = colors.HexColor('#0b0b0b'), colors.HexColor('#52514e'), colors.HexColor('#8a8984')
BLUE, ORANGE = colors.HexColor(cs.C_INSAR), colors.HexColor(cs.C_ERA5)
RULE, TINT = colors.HexColor('#e6e5e1'), colors.HexColor('#f3f2ef')
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
TEXT_W = PAGE_W - 2 * MARGIN

ss = getSampleStyleSheet()
S = {
    'title': ParagraphStyle('title', parent=ss['Title'], fontName='Helvetica-Bold', fontSize=22, leading=26,
                            textColor=INK, alignment=TA_LEFT, spaceAfter=4),
    'subtitle': ParagraphStyle('subtitle', fontName='Helvetica', fontSize=11.5, leading=15, textColor=INK_2, spaceAfter=10),
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=INK, spaceBefore=8, spaceAfter=6),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=INK, spaceBefore=6, spaceAfter=3),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=5),
    'bullet': ParagraphStyle('bullet', fontName='Helvetica', fontSize=9.6, leading=13.4, textColor=INK,
                             leftIndent=11, bulletIndent=2, spaceAfter=2.5),
    'caption': ParagraphStyle('caption', fontName='Helvetica-Oblique', fontSize=8.4, leading=11, textColor=INK_2, spaceAfter=8),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8.2, leading=11, textColor=INK_2),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.4, leading=10.5, textColor=INK),
    'cellb': ParagraphStyle('cellb', fontName='Helvetica-Bold', fontSize=8.4, leading=10.5, textColor=INK),
    'tile_v': ParagraphStyle('tile_v', fontName='Helvetica-Bold', fontSize=19, leading=22, textColor=INK),
    'tile_l': ParagraphStyle('tile_l', fontName='Helvetica', fontSize=8.2, leading=10.5, textColor=INK_2),
    'code': ParagraphStyle('code', fontName='Courier', fontSize=7.6, leading=9.8, textColor=INK, backColor=TINT,
                           borderPadding=(5, 5, 5, 5), spaceBefore=4, spaceAfter=10),
}


def p(text, style='body'):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S['bullet'], bulletText='•') for t in items]


def figure(name, caption, width=TEXT_W):
    path = os.path.join(cs.FIGURES_DIR, name + '.png')
    img = Image(path)
    img._restrictSize(width, PAGE_H * 0.62)
    return KeepTogether([img, Spacer(1, 3), p(caption, 'caption')])


def table(rows, col_widths, header_rows=1, bold_last=False):
    data = [[Paragraph(str(c), S['cellb'] if r < header_rows else S['cell']) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = [('LINEBELOW', (0, header_rows - 1), (-1, header_rows - 1), 0.8, INK_2),
             ('LINEBELOW', (0, header_rows), (-1, -1), 0.3, RULE),
             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
             ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if bold_last:
        style.append(('BACKGROUND', (0, -1), (-1, -1), TINT))
    t.setStyle(TableStyle(style))
    return t


def tiles(items):
    """Row of headline numbers: (value, label, accent colour)."""
    cells = [[Paragraph('<font color="{}">{}</font>'.format(c.hexval().replace('0x', '#'), v), S['tile_v']),
              Spacer(1, 2), Paragraph(l, S['tile_l'])] for v, l, c in items]
    w = TEXT_W / len(items)
    t = Table([cells], colWidths=[w] * len(items))
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), TINT),
                           ('LINEBEFORE', (1, 0), (-1, 0), 2, colors.white),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('LEFTPADDING', (0, 0), (-1, -1), 9), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                           ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 9)]))
    return t


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(INK_3)
    canvas.drawString(MARGIN, 10 * mm, 'InSAR4SM case study · Ford Dry Lake, California · Nov 2025 – Nov 2026')
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, 'Page {}'.format(doc.page))
    canvas.restoreState()


def fmt(v, d=1, sign=False):
    return ('{:+.%df}' % d if sign else '{:.%df}' % d).format(v)


def build():
    data = cs.load_results()
    cs.all_figures(data, save=True)
    h = cs.headline(data)
    m = cs.metrics_table(data).set_index(['tracks', 'product'])
    tr = data['tracks']
    ratio = h['rmse_era5'] / h['rmse_insar']
    story = []

    # ---------------------------------------------------------------- page 1: summary
    story += [p('InSAR4SM case study', 'small'), Spacer(1, 2),
              p('One year of satellite soil moisture in the Mojave Desert', 'title'),
              p('NISAR and Sentinel-1 soil moisture from InSAR4SM at the Ford Dry Lake SCAN station, '
                'compared with in-situ measurements and the ERA5-Land reanalysis (Nov 2025 – Nov 2026).', 'subtitle')]
    story.append(tiles([
        (fmt(h['rmse_insar']) + ' %', 'RMSE vs in situ<br/>(ERA5-Land: {} %)'.format(fmt(h['rmse_era5'])), BLUE),
        (fmt(h['r_insar'], 2), 'correlation R with in situ<br/>(ERA5-Land: {})'.format(fmt(h['r_era5'], 2)), BLUE),
        ('{:.0f} days'.format(h['median_gap_days']), 'median time between estimates<br/>({} tracks, 2 sensors)'.format(h['n_tracks']), INK),
        ('125 m', 'spacing of the soil moisture maps<br/>(ERA5-Land: ~9 km)', INK)]))
    story += [Spacer(1, 10), p('Summary', 'h1')]
    story += bullets([
        'InSAR4SM estimates surface soil moisture from SAR interferometry (coherence and phase closure), using a '
        'dielectric model, soil texture and meteorological data. It needs no in-situ calibration.',
        'Over one year at a desert site, InSAR4SM agrees with the in-situ sensor to <b>{} % RMSE (R {})</b>, '
        'against <b>{} % (R {})</b> for ERA5-Land. That is about <b>{:.1f} times less error</b>, and a lower error on every one '
        'of the {} tracks.'.format(fmt(h['rmse_insar']), fmt(h['r_insar'], 2), fmt(h['rmse_era5']), fmt(h['r_era5'], 2),
                                  ratio, h['n_tracks']),
        'NISAR (L-band) and Sentinel-1 (C-band) both work. Combining their {} tracks gives an estimate on <b>{} days</b> '
        'of the year, with a median gap of {:.0f} days and never more than {} days.'.format(
            h['n_tracks'], h['n_dates'], h['median_gap_days'], h['max_gap_days']),
        'Only the area of interest is read from each SAR product: NISAR GSLC and OPERA CSLC-S1 windows are streamed '
        'from NASA Earthdata. The full year ran in about 2.5 hours on a workstation, most of it downloading.',
    ])
    story.append(Spacer(1, 6))
    story.append(figure('fig1_overview_timeseries',
                        'Figure 1. ERA5-Land daily rain (top) and soil moisture (bottom): in-situ 5 cm sensor (black), ERA5-Land '
                        '(orange) and every InSAR4SM estimate, averaged over the 2 x 2 km area (blue circles: Sentinel-1, '
                        'green squares: NISAR).'))
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 2: site, data, method
    story.append(p('1. Site and data', 'h1'))
    story.append(table([
        ['Item', 'Description'],
        ['Site', 'Ford Dry Lake SCAN station (ISMN), 33.6547 N, -115.0976 E, Mojave Desert, California'],
        ['Climate / cover', 'Hot desert (Koppen BWh), sparse shrubland, flat terrain (slope 0.2 deg)'],
        ['Soil texture', 'Sand 79 %, clay 11 % (ISMN station metadata)'],
        ['Area of interest', '2 x 2 km square centred on the station; one estimate every 125 m (256 points), '
                             'each from a 250 x 250 m window'],
        ['Period', '2025-11-22 to 2026-11-11 (first to last NISAR acquisition)'],
        ['SAR', 'NISAR L2 GSLC (L-band, HH) and Sentinel-1 OPERA L2 CSLC-S1 (C-band, VV)'],
        ['Meteorology', 'ERA5-Land hourly precipitation, skin temperature and layer-1 (0-7 cm) soil moisture'],
        ['Reference', 'ISMN in-situ soil moisture at 5 cm, hourly (Stevens HydraProbe)'],
    ], [32 * mm, TEXT_W - 32 * mm]))
    story += [Spacer(1, 8), p('2. Processing', 'h1')]
    story += bullets([
        '<b>Data access.</b> For each track over the area, InSAR4SM finds the NISAR GSLC and OPERA CSLC-S1 products and '
        'reads only the AOI window, plus a 250 m margin. Products are already geocoded and flattened, so no SAR '
        'processing (and no ISCE) is needed. Bursts and frames are mosaicked per track.',
        '<b>Meteorology.</b> ERA5-Land precipitation over the previous 24 days identifies the dry acquisitions, which '
        'anchor the inversion. Acquisitions with frozen soil are removed.',
        '<b>Interferometric observables.</b> Per grid point: selection of distributed scatterers, coherence matrix, '
        'phase closures, and a coherence model for temporal decorrelation.',
        '<b>Inversion.</b> The soil moisture time series that best explains the observed coherence and phase closures '
        'with the De Zan interferometric model and the Hallikainen dielectric model, for the soil texture of the site.',
        '<b>Validation.</b> InSAR4SM (station cell and 2 x 2 km mean) and ERA5-Land (at the acquisition hour) against '
        'the in-situ sensor at the acquisition hour.',
    ])
    story += [Spacer(1, 6), p('Satellite tracks over the area', 'h2')]
    rows = [['Sensor', 'Track', 'Acq.', 'First', 'Last', 'Revisit (days)', 'Coh.*', 'Incidence (deg)', 'AOI covered', 'Used']]
    for _, r in tr.iterrows():
        rows.append([r.sensor, r.track, r.n_acquisitions, r['first'], r['last'], '{:.0f}'.format(r.revisit_days),
                     fmt(r.median_consecutive_coherence, 2), fmt(r.median_incLocal, 0),
                     '{:.0f} %'.format(100 * r.valid_fraction), 'yes' if r.used else 'no'])
    story.append(table(rows, [16 * mm, 14 * mm, 12 * mm, 20 * mm, 20 * mm, 18 * mm, 15 * mm, 21 * mm, 20 * mm, 14 * mm]))
    story.append(p('* Coh.: median coherence between consecutive acquisitions. NISAR track 106A only touches the edge of the area '
                   '(2 % covered) and is not used. NISAR track 157D (9 acquisitions) is below the minimum of 10 and was not '
                   'processed. Sentinel-1 173D has a 6-day revisit (Sentinel-1A and 1C).', 'caption'))
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 3: accuracy
    story.append(p('3. Accuracy against in-situ soil moisture', 'h1'))
    story.append(figure('fig3_scatter', 'Figure 2. InSAR4SM (2 x 2 km mean) and ERA5-Land against the in-situ sensor, all '
                                        'tracks pooled. Dashed: 1:1 line.', width=TEXT_W * 0.92))
    rows = [['Tracks', 'N', 'InSAR4SM<br/>RMSE (%)', 'InSAR4SM<br/>bias (%)', 'InSAR4SM<br/>R',
             'Station cell<br/>RMSE (%)', 'ERA5<br/>RMSE (%)', 'ERA5<br/>bias (%)', 'ERA5<br/>R']]
    labels = ['{} {}'.format(s, t) for s, t in data['valid_tracks']] + ['All Sentinel-1', 'All NISAR', 'All tracks']
    for lab in labels:
        a, s_, e = m.loc[(lab, 'InSAR4SM (AOI mean)')], m.loc[(lab, 'InSAR4SM (station cell)')], m.loc[(lab, 'ERA5-Land')]
        rows.append([lab, int(a.N), '<b>{}</b>'.format(fmt(a.RMSE)), fmt(a.bias, sign=True), fmt(a.R, 2),
                     fmt(s_.RMSE), fmt(e.RMSE), fmt(e.bias, sign=True), fmt(e.R, 2)])
    story.append(table(rows, [27 * mm, 10 * mm] + [(TEXT_W - 37 * mm) / 7] * 7, bold_last=True))
    story.append(p('Table 1. Scores against the in-situ 5 cm sensor at the acquisition times (soil moisture in % vol.). '
                   '"InSAR4SM" is the 2 x 2 km mean; "Station cell" is the single InSAR4SM estimate at the station; '
                   '"ERA5" is ERA5-Land layer 1.', 'caption'))
    story += bullets([
        'InSAR4SM has a lower RMSE than ERA5-Land on <b>every track</b>. The best tracks (NISAR 085D, Sentinel-1 100D) '
        'reach about 1.1 % RMSE with R 0.94-0.97.',
        'ERA5-Land is strongly biased low ({} %): its 9 km land-surface model dries out to about 0 % in early summer, '
        'while the desert soil stays at 3-5 %. InSAR4SM is only slightly low ({} %).'.format(
            fmt(m.loc[('All tracks', 'ERA5-Land')].bias, sign=True), fmt(m.loc[('All tracks', 'InSAR4SM (AOI mean)')].bias, sign=True)),
        'The single cell at the station is noisier (RMSE {} %) than the 2 x 2 km mean: a point sensor and a 250 m window '
        'see different soil. Spatial averaging gives the most robust values.'.format(fmt(m.loc[('All tracks', 'InSAR4SM (station cell)')].RMSE)),
    ])
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 4: per track time series
    story.append(p('4. Time series per track', 'h1'))
    story.append(figure('fig2_tracks_timeseries',
                        'Figure 3. One panel per track: in-situ daily mean (black line), InSAR4SM 2 x 2 km mean (connected '
                        'markers) and ERA5-Land at the same acquisition times (orange diamonds). Each title gives the scores '
                        'of the track.'))
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 5: error per track + maps
    story.append(p('5. Error per track and soil moisture maps', 'h1'))
    story.append(figure('fig4_rmse_bars', 'Figure 4. RMSE against the in-situ sensor per track and pooled.', width=TEXT_W * 0.8))
    story.append(figure('fig5_maps',
                        'Figure 5. InSAR4SM soil moisture over the 2 x 2 km area on the driest and the wettest date of the '
                        'densest track (Sentinel-1 173D, 61 acquisitions). The whole area falls inside a single ERA5-Land cell.'))
    story += bullets([
        'On the driest date the area is uniform at about 3 %. After the November 2025 rain, values range from about '
        '3 % to 20 % between neighbouring cells. Part of that is real (micro-topography, crusts, drainage), and part is '
        'noise of the per-cell inversion.',
        'Use spatially averaged values for the most accurate numbers, and the maps for patterns.',
    ])
    story.append(PageBreak())

    # ---------------------------------------------------------------- page 6: value, limitations, reproduce
    story.append(p('6. What this means for users', 'h1'))
    story += bullets([
        '<b>Accuracy:</b> about {:.1f} times lower error than ERA5-Land at this site, from satellite data only.'.format(ratio),
        '<b>Resolution:</b> estimates every 125 m, compared with ERA5-Land\'s ~9 km. Field-scale patterns of wetting and drying become visible.',
        '<b>Revisit:</b> combining NISAR and Sentinel-1 tracks gives an estimate every ~{:.0f} days.'.format(h['median_gap_days']),
        '<b>Coverage:</b> NISAR GSLC and Sentinel-1 bursts are global. In North America, OPERA CSLC-S1 removes '
        'any SAR processing on the user side (no isce3). Elsewhere, Sentinel-1 bursts are geocoded locally with COMPASS.',
        '<b>Light data handling:</b> only the area of interest is read from each product; nothing is downloaded in full.',
    ])
    story.append(p('Limitations', 'h2'))
    story += bullets([
        'One site, one year, and a hot desert with sparse vegetation: the setting where interferometric soil moisture '
        'works best. Vegetated, irrigated or frozen areas need their own validation.',
        'The reference is a single 5 cm sensor. The radar senses roughly the top 1-2 cm (C-band) to about 5 cm (L-band), '
        'so some differences are expected.',
        'The model parameters are the defaults, tuned for C-band. L-band-specific tuning may improve NISAR further.',
        'The inversion is anchored on dry acquisitions identified from ERA5-Land precipitation. Each track needs a few '
        'dry dates and about 10 or more acquisitions.',
    ])
    story.append(p('Reproduce', 'h2'))
    story.append(p('The software, this case study (notebook and result files) and the processing script are in the '
                   'InSAR4SM repository (release v2.0.0 and later): https://github.com/moistx-labs/INSAR4SM', 'body'))
    story.append(Paragraph(
        'conda env create -f INSAR4SM_env.yml &amp;&amp; conda activate insar4sm<br/>'
        'python tests/smoke/run_ford_dry_lake.py --project-dir RUN_DIR --ismn-csv ISMN_5cm.csv \\<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;--time-start 20251122T000000 --time-end 20261111T230000 --ds --min-acquisitions 10<br/>'
        'python case_studies/FordDryLake/prepare_results.py --run-dir RUN_DIR --ismn-csv ISMN_5cm.csv<br/>'
        'jupyter lab case_studies/FordDryLake/FordDryLake_case_study.ipynb', S['code']))
    story.append(p('Needs a NASA Earthdata account (~/.netrc) and a Copernicus CDS key (~/.cdsapirc). The ISMN data '
                   '(station FordDryLake, network SCAN, 5 cm) is available from https://ismn.earth after free registration.', 'small'))
    story.append(Spacer(1, 8))
    story.append(p('References', 'h2'))
    story.append(p('Karamvasis, K., &amp; Karathanassi, V. (2023). Soil moisture estimation from Sentinel-1 interferometric '
                   'observations over arid regions. Computers &amp; Geosciences, 178, 105410. '
                   'https://doi.org/10.1016/j.cageo.2023.105410', 'small'))
    story.append(Spacer(1, 3))
    story.append(p('Katsamas, G., Karamvasis, K., &amp; Karathanassi, V. (2026). Toward Improved Soil Moisture Retrieval '
                   'Using NISAR InSAR Closure Phase Measurements. EARSeL Symposium, Athens.', 'small'))
    story.append(Spacer(1, 8))
    story.append(p('Acknowledgements', 'h2'))
    logo = Image(os.path.join(cs.HERE, '..', '..', 'figures', 'moistx_logo.png'))
    logo._restrictSize(38 * mm, 12 * mm)
    ack = Table([[logo, p('The development of InSAR4SM is supported by moistX (https://github.com/moistx-labs). '
                          'The NISAR and Sentinel-1 geocoded-SLC workflow was contributed by Giorgos Katsamas.', 'small')]],
                colWidths=[44 * mm, TEXT_W - 44 * mm])
    ack.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (0, 0), 0)]))
    story.append(ack)
    story.append(Spacer(1, 8))
    story.append(p('Data credits', 'h2'))
    story.append(p('In-situ soil moisture: International Soil Moisture Network (Dorigo et al., 2021, Hydrol. Earth Syst. Sci. '
                   '25, 5749-5804), network SCAN (USDA NRCS). Meteorology: ERA5-Land (Munoz-Sabater et al., 2021, Earth Syst. '
                   'Sci. Data 13, 4349-4383), Copernicus Climate Change Service. SAR: NISAR L2 GSLC (NASA/ISRO, ASF DAAC); OPERA '
                   'L2 CSLC-S1 (NASA JPL, ASF DAAC), containing modified Copernicus Sentinel data 2025-2026. Report generated '
                   '{}.'.format(datetime.date.today().isoformat()), 'small'))

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm,
                            bottomMargin=16 * mm, title='InSAR4SM case study: Ford Dry Lake',
                            author='InSAR4SM', subject='Soil moisture validation with NISAR and Sentinel-1')
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print('written', OUT)


if __name__ == '__main__':
    build()
